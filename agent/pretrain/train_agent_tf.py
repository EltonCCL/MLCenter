"""
Parent pre-training agent class in TensorFlow.
"""

import os
import random
import numpy as np
from omegaconf import OmegaConf
import tensorflow as tf
import hydra
import logging
import wandb
from copy import deepcopy

log = logging.getLogger(__name__)
from util.scheduler_tf import CosineAnnealingWarmupRestarts  # Assuming you have TF version

def to_device(x, device='GPU:0'):
    """
    Moves the given tensor or dictionary of tensors to the specified device.

    Args:
        x (tf.Tensor or dict): The tensor or dictionary of tensors to move.
        device (str, optional): The target device. Defaults to 'GPU:0'.
    """
    assert isinstance(device, str), "Device must be a string."
    if isinstance(x, tf.Tensor):
        with tf.device(device):
            return tf.identity(x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")
        assert False, f"Unrecognized type in `to_device`: {type(x)}"

def batch_to_device(batch, device='GPU:0'):
    """
    Moves a batch of data to the specified device.

    Args:
        batch: The batch of data to move.
        device (str, optional): The target device. Defaults to 'GPU:0'.
    
    Returns:
        The batch moved to the specified device.
    """
    assert hasattr(batch, '_fields'), "Batch must have '_fields' attribute."
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)

class EMA:
    """
    Empirical Moving Average for TensorFlow models.

    Args:
        cfg: Configuration parameters containing decay rate.
    """
    def __init__(self, cfg):
        super().__init__()
        assert hasattr(cfg, 'decay'), "Config must have 'decay' parameter."
        self.beta = cfg.decay

    def update_model_average(self, ma_model, current_model):
        for current_weights, ma_weights in zip(
            current_model.trainable_weights, 
            ma_model.trainable_weights
        ):
            ma_weights.assign(
                self.update_average(ma_weights, current_weights)
            )

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class PreTrainAgent:
    """
    Parent pre-training agent class in TensorFlow.

    Args:
        cfg: Configuration parameters for initializing the agent.
    """
    def __init__(self, cfg):
        """
        Initializes the PreTrainAgent with the given configuration.

        Args:
            cfg: Configuration parameters for setting up the agent.
        """
        super().__init__()
        assert isinstance(cfg, dict) or hasattr(cfg, 'get'), "Config must be a dict or OmegaConf object."
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Set memory growth for GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        assert physical_devices, "No GPU devices found."
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Wandb
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb is not None:
            assert hasattr(cfg.wandb, 'project') and hasattr(cfg.wandb, 'run'), "Wandb config must have 'project' and 'run'."
            wandb.init(
                # entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build main model
        self.model = hydra.utils.instantiate(cfg.model)
        assert self.model is not None, "Model instantiation failed."
        self.ema = EMA(cfg.ema)
        
        # Create EMA model with identical configuration
        self.ema_model = hydra.utils.instantiate(cfg.model)
        
        # Build both models by calling them with dummy data
        batch_size = cfg.train.batch_size
        horizon_steps = cfg.horizon_steps
        obs_dim = cfg.obs_dim
        action_dim = cfg.action_dim
        cond_steps = cfg.cond_steps
        
        # Create dummy inputs in the correct format
        dummy_cond = {
            "state": tf.zeros((batch_size, cond_steps, obs_dim), dtype=tf.float32)
        }
        dummy_deterministic = True
        dummy_fixed_noise = None  # or provide specific noise if needed
        
        # Build models by calling them
        _ = self.model(dummy_cond, dummy_deterministic, dummy_fixed_noise)
        _ = self.ema_model(dummy_cond, dummy_deterministic, dummy_fixed_noise)
        assert self.model.built, "Main model is not built."
        assert self.ema_model.built, "EMA model is not built."
        
        # Copy weights from main model to EMA model
        self.ema_model.set_weights(self.model.get_weights())

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        log.info(f"Building dataset")

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        log.info("init complete")
        self.dataloader_train = (
            self.dataset_train.as_tensorflow_dataset(
                batch_size=self.batch_size,
            )
        )
        log.info(f"Finished building dataset")

        log.info(f"Building valid dataset")
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = (
                self.dataset_val.as_tensorflow_dataset()
            )
        log.info(f"Finished building valid dataset")

        log.info(f"Init optimizer")
        # Optimizer

        log.info(f"Init learning rate scheduler")
        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0
        )

        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.lr_scheduler,
            weight_decay=cfg.train.weight_decay
        )


        self.reset_parameters()
        self.epoch = 0
        log.info(f"Finished init pretrain agent")

    def run(self):
        """
        Executes the training loop for the agent.
        """
        raise NotImplementedError

    def reset_parameters(self):
        """
        Resets the parameters of the EMA model to match the main model.
        """
        # Copy weights from model to ema_model
        for target, source in zip(
            self.ema_model.trainable_variables,
            self.model.trainable_variables
        ):
            assert source is not None, "Source variable is None."
            target.assign(source)

    def step_ema(self):
        """
        Updates the EMA model based on the current model parameters.
        """
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        Saves the model and EMA to disk.
        """
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(self.epoch),
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            max_to_keep=5
        )
        save_path = manager.save(checkpoint_number=self.epoch)
        log.info(f"Saved model to {save_path}")

    def load(self, epoch):
        """
        Loads the model and EMA from disk for the specified epoch.

        Args:
            epoch (int): The epoch number to load the model from.
        """
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(0),
            model=self.model,
            ema_model=self.ema_model,
            optimizer=self.optimizer
        )
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"ckpt-{epoch}"
        )
        
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()  # Don't warn about incomplete restoration
        
        self.epoch = int(checkpoint.epoch.numpy())
        log.info(f"Loaded model from epoch {self.epoch}")