"""
Module for PPO-based Diffusion Models using TensorFlow.

This module defines the PPODiffusion class, which extends the VPGDiffusion class
to implement Proximal Policy Optimization (PPO) for diffusion-based models. It
includes methods for calculating loss, configuring the model, and reconstructing
the model from a configuration.
"""

from typing import Optional
from omegaconf import ListConfig
import tensorflow as tf
import logging
import math
import tensorflow_probability as tfp

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg_tf import VPGDiffusion


class PPODiffusion(VPGDiffusion):
    """
    Proximal Policy Optimization (PPO) Diffusion Model.

    Inherits from VPGDiffusion and implements PPO-specific loss calculations and
    configuration handling.

    Attributes:
        gamma_denoising (float): Discount factor for denoising steps.
        clip_ploss_coef (float): Clipping coefficient for policy loss.
        clip_ploss_coef_base (float): Base clipping coefficient for policy loss.
        clip_ploss_coef_rate (float): Rate at which the clipping coefficient changes.
        clip_vloss_coef (Optional[float]): Clipping coefficient for value loss.
        clip_advantage_lower_quantile (float): Lower quantile for advantage clipping.
        clip_advantage_upper_quantile (float): Upper quantile for advantage clipping.
        norm_adv (bool): Whether to normalize advantages.
    """

    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        """
        Initialize the PPODiffusion model.

        Args:
            gamma_denoising (float): Discount factor for denoising steps.
            clip_ploss_coef (float): Clipping coefficient for policy loss.
            clip_ploss_coef_base (float, optional): Base clipping coefficient for policy loss. Defaults to 1e-3.
            clip_ploss_coef_rate (float, optional): Rate at which the clipping coefficient changes. Defaults to 3.
            clip_vloss_coef (Optional[float], optional): Clipping coefficient for value loss. Defaults to None.
            clip_advantage_lower_quantile (float, optional): Lower quantile for advantage clipping. Defaults to 0.
            clip_advantage_upper_quantile (float, optional): Upper quantile for advantage clipping. Defaults to 1.
            norm_adv (bool, optional): Whether to normalize advantages. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super(PPODiffusion, self).__init__(**kwargs)

        self.norm_adv = norm_adv
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.gamma_denoising = gamma_denoising
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    @tf.function
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
        """
        Calculate the PPO loss.

        Args:
            obs (dict): Observations with keys like 'state' or 'rgb'.
                - state: Shape (B, To, Do)
                - rgb: Shape (B, To, C, H, W)
            chains_prev (Tensor): Previous chains with shape (B, K, Ta, Da).
            chains_next (Tensor): Next chains with shape (B, K, Ta, Da).
            denoising_inds (Tensor): Denoising indices with shape (B,).
            returns (Tensor): Returns with shape (B,).
            oldvalues (Tensor): Old value estimates with shape (B,).
            advantages (Tensor): Advantage estimates with shape (B,).
            oldlogprobs (Tensor): Old log probabilities with shape (B, K, Ta, Da).
            use_bc_loss (bool, optional): Whether to add behavior cloning regularization loss. Defaults to False.
            reward_horizon (int, optional): Action horizon that backpropagates gradient. Defaults to 4.

        Returns:
            Tuple: A tuple containing various loss components and statistics.
        """

        
        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )
        entropy_loss = -tf.reduce_mean(eta)

        newlogprobs = tf.clip_by_value(newlogprobs, -5.0, 2.0)
        oldlogprobs = tf.clip_by_value(oldlogprobs, -5.0, 2.0)

        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        newlogprobs = tf.reduce_mean(newlogprobs, axis=[-1, -2])
        newlogprobs = tf.reshape(newlogprobs, [-1])

        oldlogprobs = tf.reduce_mean(oldlogprobs, axis=[-1, -2])
        oldlogprobs = tf.reshape(oldlogprobs, [-1])
        bc_loss = 0.0
        if use_bc_loss:
            samples = self(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = tf.clip_by_value(bc_logprobs, -5.0, 2.0)
            bc_logprobs = tf.reduce_mean(bc_logprobs, axis=[-1, -2])
            bc_logprobs = tf.reshape(bc_logprobs, [-1])
            bc_loss = -tf.reduce_mean(bc_logprobs)
        if self.norm_adv:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
       
        advantage_min = tfp.stats.percentile(advantages, self.clip_advantage_lower_quantile * 100)
        advantage_max = tfp.stats.percentile(advantages, self.clip_advantage_upper_quantile * 100)
        advantages = tf.clip_by_value(advantages, advantage_min, advantage_max)
        steps = tf.range(tf.cast(self.ft_denoising_steps, tf.float32))
        discount = self.gamma_denoising ** (self.ft_denoising_steps - steps - 1)
        discount = tf.gather(discount, denoising_inds)
        # Then convert the list to a tensor
        # discount = tf.constant(discount_list, dtype=tf.float32)
        advantages *= discount

        logratio = newlogprobs - oldlogprobs
        ratio = tf.exp(logratio)
        t = tf.cast(denoising_inds, tf.float32) / (self.ft_denoising_steps - 1)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (tf.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t
        approx_kl = tf.reduce_mean((ratio - 1) - logratio)
        clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_ploss_coef, tf.float32))
        pg_loss1 = -tf.multiply(advantages, ratio)
        pg_loss2 = -advantages * tf.clip_by_value(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
        newvalues = self.critic(obs)
        newvalues = tf.reshape(newvalues, [-1])
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = tf.square(newvalues - returns)
            v_clipped = oldvalues + tf.clip_by_value(newvalues - oldvalues, -self.clip_vloss_coef, self.clip_vloss_coef)
            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean(v_loss_max)
        else:
            # log.info(f"newvalues: {newvalues.shape}, returns: {returns.shape}")
            # log.info(f"newvalues: {type(newvalues)}, returns: {(returns)}")
            v_loss = 0.5 * tf.reduce_mean(tf.square(newvalues - returns))
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl,
            tf.reduce_mean(ratio),
            bc_loss,
            tf.reduce_mean(eta),
        )

    def get_config(self):
        """
        Get the configuration of the PPODiffusion model.

        Returns:
            dict: Configuration dictionary including model parameters.
        """
        config = super().get_config()
        config.update({
            # ...existing config...
            "model": list(self.model) if isinstance(self.model, ListConfig) else self.model,
            "gamma_denoising": self.gamma_denoising,
            "clip_ploss_coef": self.clip_ploss_coef,
            "clip_ploss_coef_base": self.clip_ploss_coef_base,
            "clip_ploss_coef_rate": self.clip_ploss_coef_rate,
            "clip_vloss_coef": self.clip_vloss_coef,
            "clip_advantage_lower_quantile": self.clip_advantage_lower_quantile,
            "clip_advantage_upper_quantile": self.clip_advantage_upper_quantile,
            "norm_adv": self.norm_adv,
            # Add other ListConfig parameters as needed
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a PPODiffusion instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            PPODiffusion: An instance of PPODiffusion initialized with the given config.
        """
        # Convert lists back to ListConfig
        if config.get("model") is not None and isinstance(config["model"], list):
            config["model"] = ListConfig(config["model"])
        # Handle other ListConfig parameters similarly
        return cls(**config)