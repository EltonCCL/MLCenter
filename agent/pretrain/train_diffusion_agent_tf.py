"""
This module provides the TrainDiffusionAgent class which is responsible for training diffusion models using TensorFlow.
"""

import logging
import wandb
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.timer import Timer  # You need to implement or adapt this for TF
from agent.pretrain.train_agent_tf import PreTrainAgent  # The converted PreTrainAgent
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

log = logging.getLogger(__name__)

class TrainDiffusionAgent(PreTrainAgent):
    """
    TrainDiffusionAgent handles the training process for diffusion agents.

    Inherits from PreTrainAgent and integrates TensorFlow functionalities.
    """

    def __init__(self, cfg):
        """
        Initializes the TrainDiffusionAgent with the given configuration.

        Args:
            cfg: Configuration parameters for training.
        """
        super().__init__(cfg)

    @tf.function(reduce_retracing=True)
    def train_step(self, batch):
        """
        Performs a single training step on the given batch.

        Args:
            batch: A batch of training data.

        Returns:
            The computed loss for the batch.
        """

        #print input type and shape for debugging
        # print("Input batch type:", type(batch))
        # print("Input batch shape (if applicable):", [b.shape for b in batch if hasattr(b, 'shape')])

        with tf.GradientTape() as tape:
            loss = self.model.loss(*batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def run(self):
        """
        Runs the training loop for the diffusion agent.
        """
        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        log.info("Starting training")
        for epoch in range(self.n_epochs):
            # Training
            train_loss_metric = tf.keras.metrics.Mean()
            epoch_lr = self.lr_scheduler(self.epoch).numpy()
            self.optimizer.learning_rate.assign(epoch_lr)
        
            for batch_train in tqdm(self.dataloader_train):
                loss_train = self.train_step(batch_train)
                train_loss_metric.update_state(loss_train)
                
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            
            # Get accumulated loss once per epoch
            loss_train = train_loss_metric.result().numpy()
            train_loss_metric.reset_states()
            # Validation
            loss_val = None
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                loss_val_epoch = []
                for batch_val in self.dataloader_val:
                    loss_val_batch = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val_batch.numpy())
                loss_val = np.mean(loss_val_epoch)

            # Save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # Log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    metrics = {"loss - train": loss_train}
                    if loss_val is not None:
                        metrics["loss - val"] = loss_val
                    wandb.log(metrics, step=self.epoch)

            self.epoch += 1

    def save_model(self):
        """
        Saves the current model and its EMA to disk.
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
        Loads the model and its EMA from disk for the specified epoch.

        Args:
            epoch: The epoch number to load the model from.
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