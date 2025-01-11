"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

import copy
import tensorflow as tf
import logging
import numpy as np

log = logging.getLogger(__name__)

from model.diffusion.diffusion_tf import DiffusionModel, Sample
from model.diffusion.sampling_tf import make_timesteps, extract
from tensorflow_probability import distributions as tfd

class EMA:
    """
    Empirical Moving Average for TensorFlow models.

    Args:
        cfg: Configuration parameters containing decay rate.
    """
    def __init__(self, cfg):
        super().__init__()
        assert hasattr(cfg, 'decay'), "Config must have 'decay' parameter."
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

class VPGDiffusion(DiffusionModel):
    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        # eta in DDIM
        eta=None,
        learn_eta=False,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."
        log.info("---------- INIT Diffusion OK ----------")

        # Number of denoising steps to use with fine-tuned model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        assert isinstance(min_sampling_denoising_std, float)
        self.min_sampling_denoising_std = (
            tf.Variable(min_sampling_denoising_std, trainable=False)
            if isinstance(min_sampling_denoising_std, float)
            else min_sampling_denoising_std
        )

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = (
            tf.Variable(min_logprob_denoising_std, trainable=False)
            if isinstance(min_logprob_denoising_std, float)
            else min_logprob_denoising_std
        )

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta
            if not learn_eta:
                self.eta.trainable = False
                logging.info("Turned off gradients for eta")

        # Re-name network to actor
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
        assert self.actor_ft.built, "EMA model is not built."
        self.actor_ft.set_weights(self.actor.get_weights())
        self.actor_ft.trainable = True
        self.actor.trainable = False
        logging.info("Cloned and initialized the fine-tuned actor model")

        # Turn off gradients for original model
        self.actor.trainable = False
        logging.info("Turned off gradients of the pretrained network")
        finetuned_params = sum(
            [tf.size(var).numpy() for var in self.actor_ft.trainable_variables]
        )
        logging.info(f"Number of finetuned parameters: {finetuned_params}")

        # Value function
        self.critic = critic
        self.critic.trainable = True

        if network_path is not None:
            # Create a checkpoint object that includes the model and ema_model
            checkpoint = tf.train.Checkpoint(
                epoch=tf.Variable(0),  # Add epoch to the checkpoint
                model=self.actor,  # Assuming self.actor is your main model
                ema_model=self.actor_ft, # Restore ema_model's weights to actor_ft
                optimizer=tf.keras.optimizers.AdamW(
                    1e-4
                ),  # Need a dummy optimizer here as well
            )

            # Restore from the checkpoint
            manager = tf.train.CheckpointManager(
                checkpoint, network_path, max_to_keep=5
            )
            checkpoint_path = manager.latest_checkpoint

            if checkpoint_path is not None:
                # List all variables in the checkpoint
                variables = tf.train.list_variables(checkpoint_path)

                # Check for ema_model
                has_ema = False
                for var_name, _ in variables:
                    if "ema_model" in var_name:
                        has_ema = True

                if has_ema:
                    # Restore from the checkpoint
                    checkpoint.restore(checkpoint_path).expect_partial()
                    logging.info(
                        f"Loaded checkpoint from {checkpoint_path}"
                    )

                    logging.info("Assigned EMA weights to actor_ft")
                else:
                    logging.info(
                        f"Checkpoint does not contain ema_model. Loading only the main model from {checkpoint_path}"
                    )
                    # Restore only the main model
                    checkpoint = tf.train.Checkpoint(
                        model=self.actor
                    )
                    checkpoint.restore(checkpoint_path).expect_partial()

            else:
                logging.warning(
                    "No checkpoint found. Starting from scratch."
                )

        log.info("---------- INIT VPGDiffusion OK ----------")

    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        # Anneal min_sampling_denoising_std

        # if not isinstance(self.min_sampling_denoising_std, float):
        #     self.min_sampling_denoising_std.assign_next()

        # Anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            # Update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            self.actor.trainable = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

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
        use_base_policy=False,
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
        return_chain=True,
        use_base_policy=False,
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
                use_base_policy=use_base_policy,
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

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        cond_repeated = {
            key: tf.tile(
                tf.expand_dims(cond[key], axis=1),
                [1, self.ft_denoising_steps] + [1] * (cond[key].shape.ndims - 1),
            )
            for key in cond
        }
        cond_repeated = {
            key: tf.reshape(cond_repeated[key], [-1] + cond[key].shape.as_list()[1:])
            for key in cond_repeated
        }

        # Repeat t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.range(
                start=self.ft_denoising_steps - 1,
                limit=-1,
                delta=-1,
                dtype=tf.int32,
            )
            # Example: 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0

        t_all = tf.tile(t_single, [tf.shape(chains)[0]])
        if self.use_ddim:
            indices_single = tf.range(
                start=self.ddim_steps - self.ft_denoising_steps,
                limit=self.ddim_steps,
                dtype=tf.int32,
            )
            indices = tf.tile(indices_single, [tf.shape(chains)[0]])
        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        chains_prev = tf.reshape(chains_prev, [-1, self.horizon_steps, self.action_dim])
        chains_next = tf.reshape(chains_next, [-1, self.horizon_steps, self.action_dim])

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond_repeated,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, np.inf)
        dist = tfd.Normal(loc=next_mean, scale=std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    @tf.function(reduce_retracing=True)
    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains_prev: (B, Ta, Da)
            chains_next: (B, Ta, Da)
            denoising_inds: (B, )
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """
        # Sample t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.range(
                start=self.ft_denoising_steps - 1,
                limit=-1,
                delta=-1,
                dtype=tf.int32,
            )
            # Example: 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0

        t_all = tf.gather(t_single, denoising_inds)
        if self.use_ddim:
            ddim_indices_single = tf.range(
                start=self.ddim_steps - self.ft_denoising_steps,
                limit=self.ddim_steps,
                dtype=tf.int32,
            )
            ddim_indices = tf.gather(ddim_indices_single, denoising_inds)
        else:
            ddim_indices = None

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, self.min_logprob_denoising_std, np.inf)
        dist = tfd.Normal(loc=next_mean, scale=std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def loss(self, cond, chains, reward):
        assert False, "Not implemented"
        """
        REINFORCE loss. Not used right now.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            reward (to go): (B, )
        """
        # Get advantage
        value = self.critic(cond)  # (B, 1)
        advantage = reward - tf.squeeze(value, axis=-1)

        # Get logprobs for denoising steps from T-1 to 0
        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)
        # (B x K, Ta, Da)

        # Ignore obs dimension, and then sum over action dimension
        logprobs = tf.reduce_sum(logprobs[:, :, : self.action_dim], axis=-1)
        # -> (B x K, Ta)

        # -> (B, K, Ta)
        logprobs = tf.reshape(logprobs, (-1, self.denoising_steps, self.horizon_steps))

        # Sum/avg over denoising steps
        logprobs = tf.reduce_mean(logprobs, axis=1)  # -> (B, Ta)

        # Sum/avg over horizon steps
        logprobs = tf.reduce_mean(logprobs, axis=-1)  # -> (B, )

        # Get REINFORCE loss
        loss_actor = tf.reduce_mean(-logprobs * advantage)

        # Train critic to predict state value
        pred = tf.squeeze(self.critic(cond), axis=-1)
        loss_critic = tf.reduce_mean(tf.square(pred - reward))

        return loss_actor, loss_critic, eta