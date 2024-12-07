# train_diffusion_agent_tf.py

import logging
import wandb
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.timer import Timer  # You need to implement or adapt this for TF
from agent.pretrain.train_agent_tf import PreTrainAgent  # The converted PreTrainAgent

log = logging.getLogger(__name__)

class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss = self.model.loss(*batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def run(self):
        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        log.info("Starting training")
        for epoch in range(self.n_epochs):
            log.info(f"Epoch {epoch}")
            # Training
            train_loss_metric = tf.keras.metrics.Mean()
        
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

            # Update learning rate scheduler
            self.lr_scheduler.step()

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