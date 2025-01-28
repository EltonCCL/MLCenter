"""
Parent SAC fine-tuning agent class - Complete code with reward scaling (no run() method).

Adapted from train_ppo_agent.py for Soft Actor-Critic (SAC).
Reuses reward scaling from TrainPPOAgent.
"""

from typing import Optional
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts

log = logging.getLogger(__name__)
from agent.finetune.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler


class TrainSACAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Batch size for logprobs calculations after an iteration
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        self.gamma = cfg.train.gamma
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizers - SAC: Single optimizer for critics, separate for actor and alpha
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_optimizer = torch.optim.AdamW(  # Single Optimizer for both critics
            list(self.model.qf1.parameters()) + list(self.model.qf2.parameters()),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(  # Single LR Scheduler for critic optimizer
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        self.critic_tau: float = cfg.train.critic_tau
        self.update_epochs: int = cfg.train.get("update_epochs", 1)

        self.alpha: float = cfg.train.get("alpha", 0.2)
        self.automatic_entropy_tuning: bool = cfg.train.get("autotune", True)
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.train.alpha_lr)
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()

        # --------------------- REWARD SCALING (Reused from TrainPPOAgent) ---------------------
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)

        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)
        # --------------------- END REWARD SCALING ---------------------


        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_coeff: float = cfg.train.get("bc_loss_coeff", 0)


# No run() method in TrainSACAgent.py as requested.