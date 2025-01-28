# diffusion_sac.py
"""
SAC (Soft Actor-Critic) with Diffusion Policy - Corrected Twin Critic & requires_grad.
"""

import torch
import logging
import torch.nn.functional as F
from torch.distributions import Normal
import copy  # Import copy for deepcopy
import hydra # Import hydra for instantiating critic from config

from model.diffusion.diffusion_vpg import VPGDiffusion

log = logging.getLogger(__name__)


class SACDiffusion(VPGDiffusion):
    def __init__(
        self,
        critic_cfg,  # Add critic_cfg to pass critic configuration
        gamma_denoising: float,
        # SAC specific parameters
        alpha: float = 0.2,  # Entropy regularization coefficient
        automatic_entropy_tuning: bool = True,
        critic_tau: float = 0.005,  # Target network smoothing coefficient
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Discount factor for diffusion MDP (from VPGDiffusion)
        self.gamma_denoising = gamma_denoising

        # SAC specific parameters
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.critic_tau = critic_tau

        # **Corrected Critic Initialization: Two Independent Critics**
        self.qf1 = hydra.utils.instantiate(critic_cfg).to(self.device)  # Instantiate critic 1 using critic_cfg from hydra
        self.qf2 = hydra.utils.instantiate(critic_cfg).to(self.device)  # Instantiate critic 2 using critic_cfg from hydra
        self.critic_target_one = copy.deepcopy(self.qf1)  # Target network 1
        self.critic_target_two = copy.deepcopy(self.qf2)  # Target network 2

        # **Explicitly Freeze Gradients for Target Networks - BEST PRACTICE**
        for p in self.critic_target_one.parameters():
            p.requires_grad = False
        for p in self.critic_target_two.parameters():
            p.requires_grad = False

        # Target entropy for automatic temperature tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()  # Assuming continuous action space
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)  # Learnable temperature parameter
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)  # Optimizer for alpha
            self.alpha = self.log_alpha.exp().item()  # Current alpha value

    # Placeholder for GMM Entropy Estimation (Hardcoded for now)
    def estimate_entropy(self, cond, chains):
        """Placeholder for GMM entropy estimation."""
        return torch.tensor([1.0], device=self.device)

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """SAC loss for Diffusion Policy MDP."""
        # 1. Critic Loss -------------------------------------------------------
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.forward(
                cond=obs, deterministic=False, return_chain=False
            )
            # **Corrected Target Q-value Calculation: Use separate target networks**
            qf1_next_target = self.critic_target_one(obs, next_state_actions)  # Target critic 1
            qf2_next_target = self.critic_target_two(obs, next_state_actions)  # Target critic 2
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = returns + (1 - 0) * self.gamma_denoising * min_qf_next_target.reshape(-1)

        # **Corrected Current Q-value Calculation: Use separate online critics**
        qf1_a_values = self.qf1(obs, chains_prev)  # Online critic 1
        qf2_a_values = self.qf2(obs, chains_prev)  # Online critic 2
        qf1_loss = F.mse_loss(qf1_a_values.reshape(-1), next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values.reshape(-1), next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # 2. Actor Loss --------------------------------------------------------
        pi, log_pi, _ = self.forward(cond=obs, deterministic=False, return_chain=False)
        # **Corrected Actor Loss Q-value: Use both online critics**
        qf1_pi = self.qf1(obs, pi)  # Online critic 1
        qf2_pi = self.qf2(obs, pi)  # Online critic 2
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        entropy_estimate = self.estimate_entropy(obs, chains_prev)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        entropy_loss = -entropy_estimate.mean()

        # 3. Temperature Loss (Automatic Entropy Tuning) -----------------------
        if self.automatic_entropy_tuning:
            alpha_loss = - (self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = torch.tensor([0.0], device=self.device)

        alpha = self.log_alpha.exp().item() if self.automatic_entropy_tuning else self.alpha

        return (
            actor_loss,
            entropy_loss,
            qf_loss,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            alpha,
        )

    def train_step(self, loss_dict, q_optimizer, actor_optimizer, alpha_optimizer=None):
        """Perform one training step for SAC-Diffusion."""
        q_loss = loss_dict["qf_loss"]
        actor_loss = loss_dict["actor_loss"]
        alpha_loss = loss_dict["alpha_loss"]

        # Critic update - **Corrected: Optimize both critics separately**
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # Actor update
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Temperature update (if automatic entropy tuning)
        if self.automatic_entropy_tuning:
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

    def soft_update_target_networks(self, tau):
        """Soft update of the target Q-networks - **Corrected: Update both target networks**."""
        for param, target_param in zip(self.qf1.parameters(), self.critic_target_one.parameters()):  # Update target critic 1
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.critic_target_two.parameters()):  # Update target critic 2
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)