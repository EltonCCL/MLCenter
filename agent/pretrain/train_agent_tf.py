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
    if isinstance(x, tf.Tensor):
        with tf.device(device):
            return tf.identity(x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")

def batch_to_device(batch, device='GPU:0'):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)

class EMA:
    """
    Empirical moving average for TensorFlow
    """
    def __init__(self, cfg):
        super().__init__()
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
    def __init__(self, cfg):
        super().__init__()
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Set memory growth for GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # Wandb
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb  is not None:
            wandb.init(
                # entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build main model
        self.model = hydra.utils.instantiate(cfg.model)
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
            self.dataset_train.as_tensorflow_dataset()
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
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )
        
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

        self.reset_parameters()
        self.epoch = 0
        log.info(f"Finished init pretrain agent")

    def run(self):
        raise NotImplementedError

    def reset_parameters(self):
        # Copy weights from model to ema_model
        for target, source in zip(
            self.ema_model.trainable_variables,
            self.model.trainable_variables
        ):
            target.assign(source)

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_model(self):
        """
        Saves model and ema to disk
        """
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(self.epoch),
            model=self.model,
            ema_model=self.ema_model
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
        Loads model and ema from disk
        """
        checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(0),
            model=self.model,
            ema_model=self.ema_model
        )
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"ckpt-{epoch}"
        )
        
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()  # Don't warn about incomplete restoration
        
        self.epoch = int(checkpoint.epoch.numpy())
        log.info(f"Loaded model from epoch {self.epoch}")