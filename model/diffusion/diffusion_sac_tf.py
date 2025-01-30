from typing import Optional
from omegaconf import ListConfig
import tensorflow as tf
import logging
import numpy as np

log = logging.getLogger(__name__)

from model.diffusion.diffusion_tf import DiffusionModel, Sample
from model.diffusion.sampling_tf import make_timesteps, extract
from tensorflow_probability import distributions as tfd

class SACDiffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        q1,
        q2,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        # min_logprob_denoising_std=0.1,
        # eta=None,
        # learn_eta=False,
        ft_denoising_steps=0,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        log.info("---------- INIT Diffusion MLP OK ----------")
        assert not self.use_ddim, "NOT YET CHECKED WITH DDIM SUPPORT"
        assert ft_denoising_steps == 0, "FT_DENOISING_STEPS has to be 0 in SACDiffusion"
        assert ft_denoising_steps_d == 0, "FT_DENOISING_STEPS_D has to be 0 in SACDiffusion"
        assert ft_denoising_steps_t == 0, "FT_DENOISING_STEPS_T has to be 0 in SACDiffusion"
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0
        
        assert isinstance(min_sampling_denoising_std, float)
        self.min_sampling_denoising_std = (
            tf.Variable(min_sampling_denoising_std, trainable=False)
            if isinstance(min_sampling_denoising_std, float)
            else min_sampling_denoising_std
        )

        self.actor = self.network

        # Update dummy input dimensions
        dummy_input = tf.zeros([1, self.horizon_steps, self.action_dim])
        dummy_time = tf.zeros((1,))  # Usually time is a scalar per batch
        # dummy_cond = tf.zeros((1, horizon_steps, action_dim))
        dummy_cond = {
            "state": tf.zeros(
                (1, self.obs_dim)
            )  # Adjust based on actual cond structure
        }
        _ = self.actor(dummy_input, time=dummy_time, cond=dummy_cond)

        assert self.actor.built, "Main model is not built."
        self.actor.trainable = True

        self.q1 = q1
        self.q1.trainable = True
        self.q2 = q2
        self.q2.trainable = True

        # load critic
        if network_path is not None:
            # Load the checkpoint directly
            checkpoint = tf.train.load_checkpoint(network_path)
            log.info(f"NOT YET IMPLEMENT")

    def get_min_sampling_denoising_std(self):
        return self.min_sampling_denoising_std
        # if isinstance(self.min_sampling_denoising_std, float):
        #     return self.min_sampling_denoising_std
        # else:
        #     return self.min_sampling_denoising_std  # Removed .numpy()

    # Override
    @tf.function(reduce_retracing=True)
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        deterministic=False,
    ):
        """
        Optimized version using boolean masking instead of tf.tensor_scatter_nd_update.
        """
        # Dynamically determine the concatenation dimension
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = tf.where(index >= (self.ddim_steps - self.ft_denoising_steps))[
                :, 0
            ]
        else:
            ft_indices = tf.where(t < self.ft_denoising_steps)[:, 0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor

        # Overwrite noise for fine-tuning steps
        if tf.size(ft_indices) > 0:
            cond_ft = {key: tf.gather(cond[key], ft_indices) for key in cond}
            noise_ft = actor(
                tf.gather(x, ft_indices), tf.gather(t, ft_indices), cond=cond_ft
            )

            # Create a boolean mask (cast shape to int64)
            mask = tf.scatter_nd(tf.expand_dims(ft_indices, 1), tf.ones_like(ft_indices, dtype=tf.bool), shape=tf.cast(tf.shape(noise)[:1], tf.int64))
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.expand_dims(mask, axis=-1)

            # Update noise using boolean masking
            noise = tf.where(mask, noise_ft, noise)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε ) / √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:

                sqrt_recip_alphas_cumprod = extract(
                    self.sqrt_recip_alphas_cumprod, t, x.shape
                )
                sqrt_recipm1_alphas_cumprod = extract(
                    self.sqrt_recipm1_alphas_cumprod, t, x.shape
                )
                x_recon = (
                    sqrt_recip_alphas_cumprod * x - sqrt_recipm1_alphas_cumprod * noise
                )
        else:
            # Directly predicting x₀
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(
                x_recon, -self.denoised_clip_value, self.denoised_clip_value
            )
            if self.use_ddim:
                # Re-calculate noise based on clamped x_recon
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient
        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = tf.zeros((tf.shape(x)[0], 1, 1), dtype=x.dtype)
            else:
                etas = tf.expand_dims(self.eta(cond), axis=1)  # B x 1 x (Da or 1)
            sigma = tf.clip_by_value(
                etas
                * tf.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)),
                1e-10,
                np.inf,
            )
            dir_xt_coef = tf.sqrt(
                tf.clip_by_value(1.0 - alpha_prev - sigma**2, 0, np.inf)
            )
            mu = tf.sqrt(alpha_prev) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = tf.math.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁ / (1 - α̅ₜ) x₀ + √ αₜ (1 - α̅ₜ₋₁) / (1 - α̅ₜ) xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = tf.ones_like(mu)

        return mu, logvar, etas

    @tf.function(reduce_retracing=True)
    def call(
        self,
        cond,
        deterministic=False,
        return_chain=True,
    ):
        """
        Optimized version using tf.while_loop and pre-allocated chain tensor.
        """
        device = self.betas.device

        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = tf.shape(sample_data)[0]

        # Get updated minimum sampling denoising std
        # min_sampling_denoising_std = self.get_min_sampling_denoising_std()
        min_sampling_denoising_std = self.min_sampling_denoising_std
        # Initialize x with standard normal
        x = tf.random.normal((B, self.horizon_steps, self.action_dim), dtype=tf.float32)

        if self.use_ddim:
            t_all = self.ddim_t
            num_timesteps = tf.shape(self.ddim_t)[0]
        else:
            t_all = tf.reverse(tf.range(self.denoising_steps), axis=[0])
            num_timesteps = self.denoising_steps

        # Initialize loop variables
        i = tf.constant(0, dtype=tf.int32)

        # Pre-allocate chain tensor
        if return_chain:
            # Calculate max_chain_length
            if not self.use_ddim:
                max_chain_length = tf.where(self.ft_denoising_steps < self.denoising_steps, self.ft_denoising_steps + 1, self.ft_denoising_steps)
            else:
                max_chain_length = tf.where(self.ft_denoising_steps < self.ddim_steps, self.ft_denoising_steps + 1, self.ft_denoising_steps)

            chain = tf.TensorArray(dtype=tf.float32, size=max_chain_length, dynamic_size=False)

            # Conditionally initialize chain with x
            if (not self.use_ddim and self.ft_denoising_steps < self.denoising_steps) or (
                self.use_ddim and self.ft_denoising_steps < self.ddim_steps
            ):
                chain = chain.write(0, x)
                write_index = tf.constant(1, dtype=tf.int32)
            else:
                write_index = tf.constant(0, dtype=tf.int32)

        else:
            chain = tf.zeros((0,), dtype=tf.float32)
            write_index = tf.constant(0, dtype=tf.int32) # Placeholder

        def cond_fun(i, x, write_index, chain):
            return i < num_timesteps

        def body_fun(i, x, write_index, chain):
            t = t_all[i]
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                deterministic=deterministic,
            )
            std = tf.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, min_sampling_denoising_std, np.inf)
            else:
                if deterministic and t == 0:
                    std = tf.zeros_like(std)
                elif deterministic:
                    std = tf.clip_by_value(std, 1e-3, np.inf)
                else:
                    std = tf.clip_by_value(std, min_sampling_denoising_std, np.inf)
            noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            noise = tf.clip_by_value(
                noise, -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # Clamp action at final step
            if self.final_action_clip_value is not None and i == num_timesteps - 1:
                x = tf.clip_by_value(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            # Conditionally write to chain using write_index
            if return_chain:
                # Create a mask for when to write to the chain
                if self.use_ddim:
                    write_condition = i >= (self.ddim_steps - self.ft_denoising_steps)
                else:
                    write_condition = t < self.ft_denoising_steps

                # Write to chain and increment write_index if condition is met
                chain = tf.cond(write_condition, lambda: chain.write(write_index, x), lambda: chain)
                write_index = tf.cond(write_condition, lambda: write_index + 1, lambda: write_index)

            return i + 1, x, write_index, chain

        # Use tf.while_loop for iteration
        _, x, write_index, chain = tf.while_loop(
            cond_fun,
            body_fun,
            [i, x, write_index, chain],
            shape_invariants=[
                i.get_shape(),
                x.get_shape(),
                write_index.get_shape(),
                tf.TensorShape(None), # Use TensorArray instead
            ],
        )

        # Process the chain after the loop
        if return_chain:
            chain = chain.stack()
            chain = tf.transpose(chain, perm=[1, 0, 2, 3])

        return Sample(x, chain)