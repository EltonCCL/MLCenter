import math
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
        ft_denoising_steps,
        learn_alpha=True,
        network_path=None,
        min_sampling_denoising_std=0.1,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        cond_dim=1,
        action_step=1,
        # DACER SPECIFIC
        logalpha=math.log(3),
        lambda_=0.1,
        entropy=0.0,
        target_entropy=-0.9,
        tau=0.005,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not self.use_ddim, "NOT YET CHECKED WITH DDIM SUPPORT"
        log.info("---------- INIT Diffusion MLP OK ----------")
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

        self.cond_dim = cond_dim
        self.action_step = action_step
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

        # Clone self.actor to self.actor_tf
        self.actor_ft = tf.keras.models.clone_model(self.actor)

        # Build the cloned model by calling it with dummy inputs
        _ = self.actor_ft(dummy_input, time=dummy_time, cond=dummy_cond)
        assert self.actor.built, "Main model is not built."
        assert self.actor_ft.built, "ft model is not built."
        self.actor_ft.set_weights(self.actor.get_weights())
        self.actor_ft.trainable = True
        self.actor.trainable = False
        log.info("Cloned and initialized the fine-tuned actor model")

        # Turn off gradients for original model
        self.actor.trainable = False
        log.info("Turned off gradients of the pretrained network")
        finetuned_params = sum(
            [tf.size(var).numpy() for var in self.actor_ft.trainable_variables]
        )

        # log.info(f"Number of finetuned parameters: {finetuned_params}")

        self.q1 = q1
        self.q1.trainable = True
        self.q2 = q2
        self.q2.trainable = True

        dummy_obs = {'state': tf.zeros((1, self.cond_dim))}
        dummy_action = tf.zeros((1, self.action_dim * self.action_step))
        
        _ = self.q1(dummy_obs, dummy_action)
        _ = self.q2(dummy_obs, dummy_action)
        assert self.q1.built, "q1 is not built."
        assert self.q2.built, "q2 is not built."
        
        self.q1_target = tf.keras.models.clone_model(self.q1)
        self.q2_target = tf.keras.models.clone_model(self.q2)
        self.q1_target.trainable = True
        self.q2_target.trainable = True

        _ = self.q1_target(dummy_obs, dummy_action)
        _ = self.q2_target(dummy_obs, dummy_action)
        assert self.q1_target.built, "q1 is not built."
        assert self.q2_target.built, "q2 is not built."

        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

        # DACER specific
        self.logalpha = tf.Variable(logalpha, trainable=learn_alpha)
        self.lambda_ = tf.constant(lambda_)
        self.entropy = tf.constant(entropy, dtype=tf.float32)
        self.target_entropy = tf.constant(target_entropy * self.action_dim, dtype=tf.float32)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.learn_alpha = learn_alpha


        if network_path is not None:
            checkpoint_vars = {}
            has_ft = False
            for name, shape in tf.train.list_variables(network_path):
                    checkpoint_vars[name] = shape
                    if "ft_model" in name:
                        has_ft = True
            
            log.info(f"Found fine-tuned model in checkpoint: {has_ft}")
            if has_ft:
                checkpoint = tf.train.Checkpoint(
                    itr=tf.Variable(0),
                    ft_model=tf.train.Checkpoint(
                        network=self.actor,
                        actor_ft=self.actor_ft,
                        min_sampling_denoising_std=self.min_sampling_denoising_std,
                        q1=self.q1,
                        q2=self.q2,
                        q1_target=self.q1_target,
                        q2_target=self.q2_target,
                        logalpha=self.logalpha,
                    )
                )

                initial_actor_params = self.actor.get_weights()
                initial_actor_ft_params = self.actor_ft.get_weights()
                initial_q1_params = self.q1.get_weights()
                initial_q2_params = self.q2.get_weights()
                initial_q1_target_params = self.q1_target.get_weights()
                initial_q2_target_params = self.q2_target.get_weights()
                initial_logalpha = self.logalpha.numpy()
                initial_min_sampling_denoising_std = self.min_sampling_denoising_std.numpy()

                status = checkpoint.restore(network_path)

                log.info(f"Restored fine-tuned model from {network_path}")
                loaded_actor_params = self.actor.get_weights()
                loaded_actor_ft_params = self.actor_ft.get_weights()
                loaded_q1_params = self.q1.get_weights()
                loaded_q2_params = self.q2.get_weights()
                loaded_q1_target_params = self.q1_target.get_weights()
                loaded_q2_target_params = self.q2_target.get_weights()
                loaded_logalpha = self.logalpha.numpy()
                loaded_min_sampling_denoising_std = self.min_sampling_denoising_std.numpy()

                assert len(initial_actor_params) == len(loaded_actor_params), "Actor parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_actor_params, loaded_actor_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Actor parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Actor parameters are the same after loading (This is expected for non-ft actor if not saved in ckpt)" # Expect same for non-ft actor

                # Actor FT
                assert len(initial_actor_ft_params) == len(loaded_actor_ft_params), "Actor FT parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_actor_ft_params, loaded_actor_ft_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Actor FT parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Actor FT parameters are the same"

                # Q1
                assert len(initial_q1_params) == len(loaded_q1_params), "Q1 parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_q1_params, loaded_q1_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Q1 parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Q1 parameters are the same"

                # Q2
                assert len(initial_q2_params) == len(loaded_q2_params), "Q2 parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_q2_params, loaded_q2_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Q2 parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Q2 parameters are the same"

                # Q1 Target
                assert len(initial_q1_target_params) == len(loaded_q1_target_params), "Q1 Target parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_q1_target_params, loaded_q1_target_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Q1 Target parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Q1 Target parameters are the same"

                # Q2 Target
                assert len(initial_q2_target_params) == len(loaded_q2_target_params), "Q2 Target parameters mismatch"
                for original_parameter, loaded_parameter in zip(initial_q2_target_params, loaded_q2_target_params):
                    assert original_parameter.shape == loaded_parameter.shape, "Q2 Target parameter shape mismatch"
                    assert not np.array_equal(original_parameter, loaded_parameter), "Q2 Target parameters are the same"

                # logalpha
                assert initial_logalpha.shape == loaded_logalpha.shape, "logalpha shape mismatch"

                if np.array_equal(initial_logalpha, loaded_logalpha):
                    log.warning(f"Previous Alpha: {initial_logalpha}")
                    log.warning(f"Loaded Alpha: {loaded_logalpha}")
                    log.warning("logalpha is the same after loading. Please check if this is expected. (i.e. you want the logalpha to be a constant)")
                # assert not np.array_equal(initial_logalpha, loaded_logalpha), "logalpha is the same"

                # min_sampling_denoising_std
                assert initial_min_sampling_denoising_std.shape == loaded_min_sampling_denoising_std.shape, "min_sampling_denoising_std shape mismatch"
                if np.array_equal(initial_min_sampling_denoising_std, loaded_min_sampling_denoising_std):
                    log.warning("min_sampling_denoising_std is the same after loading. Please check if this is expected. (i.e. you want the min_sampling_denoising_std to be a constant)")
        
    
    def update_target(self):
        for target_param, param in zip(self.q1_target.trainable_variables, self.q1.trainable_variables):
            target_param.assign(param * self.tau + (1.0 - self.tau) * target_param)
        for target_param, param in zip(self.q2_target.trainable_variables, self.q2.trainable_variables):
            target_param.assign(param * self.tau + (1.0 - self.tau) * target_param)

    def get_min_sampling_denoising_std(self):
        return self.min_sampling_denoising_std

    # Override
    @tf.function(reduce_retracing=True)
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        """
        Optimized version using boolean masking instead of tf.tensor_scatter_nd_update.
        """
        # Dynamically determine the concatenation dimension
        noise = self.actor_ft(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = tf.where(index >= (self.ddim_steps - self.ft_denoising_steps))[
                :, 0
            ]
        else:
            ft_indices = tf.where(t < self.ft_denoising_steps)[:, 0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft

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
        return_chain=False,
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

            # # DACER approach
            # noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
            # x = x + noise * tf.math.exp(self.logalpha) * self.lambda_

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

        # DACER approach
        noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
        x = x + noise * tf.math.exp(self.logalpha) * self.lambda_

        # Clamp action at final step
        if self.final_action_clip_value is not None:
            x = tf.clip_by_value(
                x, -self.final_action_clip_value, self.final_action_clip_value
            )

        # Process the chain after the loop
        if return_chain:
            chain = chain.stack()
            chain = tf.transpose(chain, perm=[1, 0, 2, 3])

        return Sample(x, chain)
