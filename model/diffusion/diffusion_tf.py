"""
Gaussian Diffusion Model implementation using TensorFlow.

This module provides the DiffusionModel class, which encapsulates the Gaussian
Diffusion process for training and sampling in reinforcement learning environments.
"""

import logging
import tensorflow as tf
import numpy as np
from collections import namedtuple
from omegaconf import ListConfig

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")

from model.diffusion.sampling_tf import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

class DiffusionModel(tf.keras.Model):
    """
    Gaussian Diffusion Model with optional DDIM sampling for TensorFlow.

    Args:
        network (tf.keras.Model): The neural network model.
        horizon_steps (int): Number of horizon steps.
        obs_dim (int): Dimension of observations.
        action_dim (int): Dimension of actions.
        network_path (str, optional): Path to the network weights.
        device (str, optional): Device to run the model on.
        denoised_clip_value (float, optional): Clipping value for denoised output.
        randn_clip_value (float, optional): Clipping value for random noise.
        final_action_clip_value (float, optional): Clipping value for final actions.
        eps_clip_value (float, optional): Clipping value for epsilon (DDIM only).
        denoising_steps (int, optional): Number of denoising steps.
        predict_epsilon (bool, optional): Whether to predict epsilon.
        use_ddim (bool, optional): Whether to use DDIM sampling.
        ddim_discretize (str, optional): DDIM discretization method.
        ddim_steps (int, optional): Number of DDIM steps.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="/GPU:0",
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,
        denoising_steps=100,
        predict_epsilon=True,
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        self.denoised_clip_value = denoised_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.randn_clip_value = randn_clip_value
        self.eps_clip_value = eps_clip_value

        # Initialize DDPM parameters *before* network building
        with tf.device(device):
            self._initialize_ddpm_params()
            self.network = network

        # Now proceed with network loading
        if network_path is not None:
            # Create dummy inputs for building the network
            dummy_input = tf.zeros((1, horizon_steps, action_dim))
            dummy_time = tf.zeros((1,))
            dummy_cond = {"state": tf.zeros((1, self.obs_dim))}

            # Call the network with dummy inputs to build it
            with tf.device(device):
                self.network(dummy_input, time=dummy_time, cond=dummy_cond)

            checkpoint = tf.train.Checkpoint(model=self.network)
            manager = tf.train.CheckpointManager(
                checkpoint, network_path, max_to_keep=5
            )
            if manager.latest_checkpoint:
                checkpoint.restore(manager.latest_checkpoint).expect_partial()
                log.info(f"Loaded model from {manager.latest_checkpoint}")
            else:
                log.warning(f"No checkpoint found at {network_path}")


    # @tf.function
    def _initialize_ddpm_params(self):
        """
        Initialize DDPM parameters as a tf.function.
        """
        # DDPM parameters
        self.betas = cosine_beta_schedule(self.denoising_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = tf.concat(
            [tf.ones(1), self.alphas_cumprod[:-1]], axis=0
        )
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)

        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = tf.math.log(
            tf.clip_by_value(self.ddpm_var, 1e-20, float("inf"))
        )

        self.ddpm_mu_coef1 = (
            self.betas * tf.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * tf.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        if self.use_ddim:
            assert self.predict_epsilon, "DDIM requires predicting epsilon for now."
            if self.ddim_discretize == "uniform":
                step_ratio = self.denoising_steps // self.ddim_steps
                self.ddim_t = tf.range(0, self.ddim_steps) * step_ratio
            else:
                raise ValueError("Unknown discretization method for DDIM.")

            # Use TensorArray to store intermediate tensors
            ddim_alphas_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            ddim_alphas_sqrt_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            ddim_alphas_prev_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            ddim_sqrt_one_minus_alphas_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            ddim_sigmas_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )
            ddim_t_ta = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False
            )

            for i in tf.range(self.ddim_steps):
                ddim_alphas_ta = ddim_alphas_ta.write(
                    i, tf.gather(self.alphas_cumprod, tf.cast(self.ddim_t[i], tf.int32))
                )

                ddim_alphas_sqrt_ta = ddim_alphas_sqrt_ta.write(
                    i,
                    tf.sqrt(
                        tf.gather(self.alphas_cumprod, tf.cast(self.ddim_t[i], tf.int32))
                    ),
                )

                ddim_sqrt_one_minus_alphas_ta = ddim_sqrt_one_minus_alphas_ta.write(
                    i,
                    tf.pow(
                        1.0
                        - tf.gather(
                            self.alphas_cumprod, tf.cast(self.ddim_t[i], tf.int32)
                        ),
                        0.5,
                    ),
                )
                ddim_t_ta = ddim_t_ta.write(i, self.ddim_t[i])

            ddim_alphas_prev_ta = ddim_alphas_prev_ta.write(0, 1.0)
            for i in tf.range(1, self.ddim_steps):
                ddim_alphas_prev_ta = ddim_alphas_prev_ta.write(
                    i, tf.gather(self.alphas_cumprod, tf.cast(self.ddim_t[i - 1], tf.int32))
                )

            ddim_eta = 0
            for i in tf.range(self.ddim_steps):
                ddim_sigmas_ta = ddim_sigmas_ta.write(
                    i,
                    ddim_eta
                    * tf.pow(
                        (
                            1
                            - tf.gather(
                                self.alphas_cumprod, tf.cast(self.ddim_t[i] - 1, tf.int32)
                            )
                        )
                        / (
                            1
                            - tf.gather(
                                self.alphas_cumprod, tf.cast(self.ddim_t[i], tf.int32)
                            )
                        )
                        * (
                            1
                            - tf.gather(
                                self.alphas_cumprod, tf.cast(self.ddim_t[i], tf.int32)
                            )
                            / tf.gather(
                                self.alphas_cumprod, tf.cast(self.ddim_t[i] - 1, tf.int32)
                            )
                        ),
                        0.5,
                    ),
                )

            # Flip all
            reverse_indices = tf.range(self.ddim_steps - 1, -1, -1)

            self.ddim_t = tf.cast(
                tf.gather(ddim_t_ta.stack(), reverse_indices), tf.int32
            )
            self.ddim_alphas = tf.gather(ddim_alphas_ta.stack(), reverse_indices)
            self.ddim_alphas_sqrt = tf.gather(ddim_alphas_sqrt_ta.stack(), reverse_indices)
            self.ddim_alphas_prev = tf.gather(ddim_alphas_prev_ta.stack(), reverse_indices)
            self.ddim_sqrt_one_minus_alphas = tf.gather(
                ddim_sqrt_one_minus_alphas_ta.stack(), reverse_indices
            )
            self.ddim_sigmas = tf.gather(ddim_sigmas_ta.stack(), reverse_indices)

    def get_config(self):
        """
        Returns the configuration of the DiffusionModel.

        Converts ListConfig to list if necessary.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "network": self.network,
                "horizon_steps": self.horizon_steps,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "network_path": self.network_path,
                "device": self.device,
                "denoised_clip_value": self.denoised_clip_value,
                "randn_clip_value": self.randn_clip_value,
                "final_action_clip_value": self.final_action_clip_value,
                "eps_clip_value": self.eps_clip_value,
                "denoising_steps": self.denoising_steps,
                "predict_epsilon": self.predict_epsilon,
                "use_ddim": self.use_ddim,
                "ddim_discretize": self.ddim_discretize,
                "ddim_steps": self.ddim_steps,
                # Convert ListConfig to list if necessary
                "some_list_param": (
                    list(self.some_list_param)
                    if hasattr(self, "some_list_param")
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a DiffusionModel instance from a configuration dictionary.

        Converts lists back to ListConfig if needed.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            DiffusionModel: An instance of DiffusionModel.
        """
        # Convert lists back to ListConfig if needed
        if config.get("some_list_param") is not None:
            config["some_list_param"] = ListConfig(config["some_list_param"])
        return cls(**config)

    @tf.function
    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        """
        Compute the mean and variance for the denoising step.

        Args:
            x (Tensor): Current sample.
            t (Tensor): Timesteps.
            cond (dict): Conditioning information.
            index (Tensor, optional): Index for DDIM.
            network_override (callable, optional): Override network function.

        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance.
        """
        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond)
        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(
                x_recon, -self.denoised_clip_value, self.denoised_clip_value
            )
            if self.use_ddim:
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        if self.use_ddim:
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = tf.sqrt(1.0 - alpha_prev - tf.square(sigma)) * noise
            mu = tf.sqrt(alpha_prev) * x_recon + dir_xt
            var = tf.square(sigma)
            logvar = tf.math.log(var)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)

        return mu, logvar

    @tf.function
    def call(self, cond, deterministic=True, fixed_noise=None):
        """
        Forward pass for sampling actions.

        Args:
            cond (dict): Conditioning dictionary with observations.
            deterministic (bool, optional): Whether to use deterministic sampling.
            fixed_noise (Tensor, optional): Fixed noise for reproducibility.

        Returns:
            Sample: Named tuple containing trajectories.
        """
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = tf.shape(sample_data)[0]

        if fixed_noise is not None:
            x = fixed_noise
        else:
            x = tf.random.normal([B, self.horizon_steps, self.action_dim])

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = tf.range(self.denoising_steps - 1, -1, -1)

        # Use tf.while_loop instead of Python for loop
        i = tf.constant(0, dtype=tf.int32)
        x_shape = tf.shape(x)

        def cond_fun(i, x):
            return i < tf.shape(t_all)[0]

        def body_fun(i, x):
            t = t_all[i]
            t_b = make_timesteps(B, t, self.betas.device)
            index_b = make_timesteps(B, i, self.betas.device)

            mean, logvar = self.p_mean_var(x=x, t=t_b, cond=cond, index=index_b)
            std = tf.exp(0.5 * logvar)

            if self.use_ddim:
                std = tf.zeros_like(std)
            else:
                # Use boolean masking for t == 0 condition
                mask = tf.cast(tf.not_equal(t, 0), tf.float32)
                std = std * mask + (1.0 - mask) * tf.zeros_like(std) # when t==0, std = 0
                std = tf.clip_by_value(std, 1e-3, float("inf"))

            if fixed_noise is not None:
                noise = tf.clip_by_value(
                    fixed_noise, -self.randn_clip_value, self.randn_clip_value
                )
            else:
                noise = tf.clip_by_value(
                    tf.random.normal(x_shape),
                    -self.randn_clip_value,
                    self.randn_clip_value,
                )
            x = mean + std * noise

            if self.final_action_clip_value is not None:
                x = tf.where(
                    tf.equal(i, tf.shape(t_all)[0] - 1),
                    tf.clip_by_value(
                        x, -self.final_action_clip_value, self.final_action_clip_value
                    ),
                    x,
                )

            return i + 1, x

        _, x = tf.while_loop(cond_fun, body_fun, loop_vars=[i, x])

        return Sample(x, None)

    @tf.function
    def loss(self, x, *args):
        """
        Compute the loss for training.

        Args:
            x (Tensor): Input data.
            *args: Additional arguments.

        Returns:
            Tensor: Loss value.
        """
        batch_size = tf.shape(x)[0]
        t = tf.random.uniform([batch_size], 0, self.denoising_steps, dtype=tf.int32)
        return self.p_losses(x, *args, t)

    @tf.function
    def p_losses(self, x_start, cond, t):
        """
        Compute the denoising loss.

        Args:
            x_start (Tensor): Original data.
            cond (dict): Conditioning information.
            t (Tensor): Timesteps.

        Returns:
            Tensor: Loss value.
        """
        noise = tf.random.normal(tf.shape(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.network(x_noisy, t, cond)

        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))

    @tf.function
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process.

        Args:
            x_start (Tensor): Original data.
            t (Tensor): Timesteps.
            noise (Tensor, optional): Noise to add.

        Returns:
            Tensor: Noisy data.
        """
        if noise is None:
            noise = tf.random.normal(tf.shape(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )