"""Module for training a PPO agent using TensorFlow."""

from typing import Optional
import tensorflow as tf
import logging
from util.scheduler_tf import (
    CosineAnnealingWarmupRestarts,
)  # Implement a TensorFlow equivalent
from agent.finetune.train_agent_tf import TrainAgent  # Updated base class
from util.reward_scaling import RunningRewardScaler

log = logging.getLogger(__name__)


class TrainPPOAgent(TrainAgent):
    """TrainPPOAgent class for fine-tuning PPO agents with TensorFlow.

    Args:
        cfg: Configuration object containing training parameters.
    """

    def __init__(self, cfg):
        """Initialize the TrainPPOAgent with the given configuration.

        Args:
            cfg: Configuration object containing training parameters.
        """
        super().__init__(cfg)

        # Batch size for logprobs calculations after an iteration
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # Discount factor gamma
        self.gamma = cfg.train.gamma

        # Warmup iterations for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Cosine scheduler with linear warmup
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.actor_lr_scheduler,
            weight_decay=cfg.train.actor_weight_decay,  # TensorFlow uses decay differently
        )

        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )

        # Generalized Advantage Estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # Target KL divergence
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of gradient updates per iteration
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef: float = cfg.train.get("ent_coef", 0)

        # Value loss coefficient
        self.vf_coef: float = cfg.train.get("vf_coef", 0)

        # Running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)

        # Reward scaling constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)

        # Behavior cloning (if used)
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_coeff: float = cfg.train.get("bc_loss_coeff", 0)

    def reset_actor_optimizer(self):
        """Reset the actor optimizer to its initial state."""
        # Save the current optimizer weights
        old_weights = self.actor_optimizer.get_weights()

        # Create a new scheduler
        new_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.actor_lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Create a new optimizer with the new scheduler
        new_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.cfg.train.actor_lr,
            decay=self.cfg.train.actor_weight_decay,
        )

        # Set the new optimizer's weights to the saved weights
        new_optimizer.set_weights(old_weights)

        # Assign the new optimizer
        self.actor_optimizer = new_optimizer

        log.info("Reset actor optimizer")
