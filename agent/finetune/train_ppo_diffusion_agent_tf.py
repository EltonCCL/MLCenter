"""
Module for training PPO Diffusion agents using TensorFlow.
"""

import os
import pickle
import einops
import numpy as np
import tensorflow as tf
import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer  # Implement or adapt for TensorFlow compatibility
from agent.finetune.train_ppo_agent_tf import TrainPPOAgent  # Updated base class
from util.scheduler_tf import CosineAnnealingWarmupRestarts  # TensorFlow scheduler


class TrainPPODiffusionAgent(TrainPPOAgent):
    """
    Trainer for PPO Diffusion Agents.

    Args:
        cfg (dict): Configuration dictionary containing training parameters.
    """

    def __init__(self, cfg):
        """
        Initialize the TrainPPODiffusionAgent.

        Args:
            cfg (dict): Configuration dictionary containing training parameters.
        """
        super().__init__(cfg)
        self.n_cond_step = cfg.cond_steps  # Ensure cond_steps is correctly set

        # Reward horizon configuration
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta parameter between DDIM and DDPM
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                initial_learning_rate=cfg.train.eta_lr,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
                n_critic_warmup_itr=self.n_critic_warmup_itr,
            )
            self.eta_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.eta_lr_scheduler,
                weight_decay=cfg.train.eta_weight_decay,
            )

    def run(self):
        """
        Execute the training loop for the PPO Diffusion agent.

        Args:
            None
        """
        # Start training loop
        timer = Timer()  # Ensure compatibility or replace with TensorFlow's timer
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            # Prepare video paths for each environment
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Determine if the current iteration is for evaluation
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train

            # Reset environments before iteration starts if needed
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            # Initialize trajectory holders
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Collect trajectories from environments
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action based on current observations
                if not eval_mode:
                    tf.keras.backend.clear_session()  # Optional: clear session if needed
                cond = {
                    "state": tf.convert_to_tensor(
                        prev_obs_venv["state"], dtype=tf.float32
                    )
                }
                samples = self.model(
                    cond=cond,
                    deterministic=eval_mode,
                    return_chain=True,
                    training=not eval_mode,  # Indicate training mode
                )
                output_venv = (
                    samples.trajectories.numpy()
                )  # Convert TensorFlow tensor to numpy
                chains_venv = samples.chains.numpy()
                action_venv = output_venv[:, : self.act_steps]

                # Apply selected actions to the environments
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                if self.save_full_observations:
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # Update observations for the next step
                prev_obs_venv = obs_venv

                # Increment training step counter
                if not eval_mode:
                    cnt_train_step += self.n_envs * self.act_steps

            # Summarize episode rewards
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if self.furniture_sparse_reward:
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.warning("No episode completed within the iteration!")

            # Update models if not in evaluation mode
            if not eval_mode:
                # Convert trajectory data to tensors
                obs_trajs["state"] = tf.convert_to_tensor(
                    obs_trajs["state"], dtype=tf.float32
                )

                # Calculate value estimates and log probabilities in batches
                num_split = math.ceil(
                    self.n_envs * self.n_steps / self.logprob_batch_size
                )
                obs_ts = [{} for _ in range(num_split)]
                obs_k = einops.rearrange(
                    obs_trajs["state"],
                    "s e ... -> (s e) ...",
                )
                total_size = obs_k.shape[0]
                num_splits = math.ceil(total_size / self.logprob_batch_size)
                size_splits = [self.logprob_batch_size] * (num_splits - 1)
                size_splits.append(
                    total_size - self.logprob_batch_size * (num_splits - 1)
                )

                obs_ts_k = tf.split(obs_k, size_splits, axis=0)
                for i, obs_t in enumerate(obs_ts_k):
                    obs_ts[i]["state"] = obs_t
                values_trajs = []
                for obs in obs_ts:
                    values = self.model.critic(obs).numpy().flatten()
                    values_trajs.append(values.reshape(-1, self.n_envs))
                values_trajs = np.vstack(values_trajs)

                chains_t = einops.rearrange(
                    tf.convert_to_tensor(chains_trajs, dtype=tf.float32),
                    "s e t h d -> (s e) t h d",
                )
                chains_ts_k = tf.split(chains_t, size_splits, axis=0)

                logprobs_trajs = []
                for obs, chains in zip(obs_ts, chains_ts_k):
                    logprobs = self.model.get_logprobs(obs, chains).numpy()
                    # Preserve the denoising_steps dimension in reshape
                    logprobs_trajs.append(
                        logprobs.reshape(1, *logprobs.shape)
                    )  # Keep all dimensions
                logprobs_trajs = np.vstack(logprobs_trajs)
                log.info(f"obs_ts: {len(obs_ts)} {obs_ts[0]['state'].shape}")
                log.info(f"chains_ts_k: {len(chains_ts_k)} {chains_ts_k[0].shape}")
                log.info(f"logprobs_trajs: {logprobs_trajs.shape}")

                # Normalize rewards if enabled
                if self.reward_scale_running:
                    reward_trajs_transpose = self.running_reward_scaler(
                        reward=reward_trajs.T, first=firsts_trajs[:-1].T
                    )
                    reward_trajs = reward_trajs_transpose.T

                # Generalized Advantage Estimation (GAE) calculation
                obs_venv_ts = {
                    "state": tf.convert_to_tensor(obs_venv["state"], dtype=tf.float32)
                }
                advantages_trajs = np.zeros_like(reward_trajs)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextvalues = (
                            self.model.critic(obs_venv_ts).numpy().reshape(1, -1)
                        )
                    else:
                        nextvalues = values_trajs[t + 1]
                    nonterminal = 1.0 - terminated_trajs[t]
                    delta = (
                        reward_trajs[t] * self.reward_scale_const
                        + self.gamma * nextvalues * nonterminal
                        - values_trajs[t]
                    )
                    advantages_trajs[t] = (
                        delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                    )
                    lastgaelam = advantages_trajs[t]
                returns_trajs = advantages_trajs + values_trajs

                # Prepare tensors for gradient updates
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    tf.convert_to_tensor(chains_trajs, dtype=tf.float32),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = tf.reshape(
                    tf.convert_to_tensor(returns_trajs, dtype=tf.float32), [-1]
                )
                values_k = tf.reshape(
                    tf.convert_to_tensor(values_trajs, dtype=tf.float32), [-1]
                )
                advantages_k = tf.reshape(
                    tf.convert_to_tensor(advantages_trajs, dtype=tf.float32), [-1]
                )
                logprobs_k = tf.convert_to_tensor(logprobs_trajs, dtype=tf.float32)
                log.info(f"logprobs_k: {logprobs_k.shape}")

                # Initialize variables for tracking
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []

                # Begin update epochs
                for update_epoch in range(self.update_epochs):
                    # Shuffle the data indices for this epoch
                    inds_k = tf.random.shuffle(tf.range(total_steps))
                    log.info(f"total_steps: {total_steps}")
                    num_batch = max(1, total_steps // self.batch_size)
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]
                        log.info(f"inds_k: {inds_k.shape}")
                        log.info(f"start: {start}")
                        log.info(f"end: {end}")

                        # Unravel indices for batching
                        batch_inds_b, denoising_inds_b = tf.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        log.info(f"inds_b: {inds_b.shape}")
                        log.info(f"self.n_steps: {self.n_steps}")
                        log.info(f"self.n_envs: {self.n_envs}")
                        log.info(
                            f"self.model.ft_denoising_steps: {self.model.ft_denoising_steps}"
                        )

                        # Gather observations for the current batch
                        log.info(f"obs_k: {obs_k['state'].shape}")
                        log.info(f"batch_inds_b: {batch_inds_b.shape}")
                        obs_b = {"state": tf.gather(obs_k["state"], batch_inds_b)}

                        # Use advanced indexing to gather chains
                        indices = tf.stack([batch_inds_b, denoising_inds_b], axis=1)
                        chains_prev_b = tf.gather_nd(chains_k, indices)

                        indices_next = tf.stack(
                            [batch_inds_b, denoising_inds_b + 1], axis=1
                        )
                        chains_next_b = tf.gather_nd(chains_k, indices_next)

                        # Gather additional required tensors
                        returns_b = tf.gather(returns_k, batch_inds_b)
                        values_b = tf.gather(values_k, batch_inds_b)
                        advantages_b = tf.gather(advantages_k, batch_inds_b)

                        # Gather log probabilities with both indices
                        logprobs_b = tf.gather_nd(logprobs_k, indices)

                        with tf.GradientTape() as tape:
                            # Compute losses
                            log.info(f"chains_k: {chains_k.shape}")  # same
                            log.info(f"obs_b: {obs_b['state'].shape}")  # not
                            log.info(f"returns_k: {returns_k.shape}")
                            log.info(f"values_k: {values_k.shape}")
                            log.info(f"advantages_k: {advantages_k.shape}")
                            log.info(f"logprobs_b: {logprobs_b.shape}")
                            log.info(f"chains_prev_b: {chains_prev_b.shape}")
                            log.info(f"chains_next_b: {chains_next_b.shape}")
                            log.info(f"denoising_inds_b: {denoising_inds_b.shape}")
                            log.info(f"returns_b: {returns_b.shape}")
                            log.info(f"values_b: {values_b.shape}")
                            log.info(f"advantages_b: {advantages_b.shape}")
                            log.info(f"logprobs_b: {logprobs_b.shape}")
                            (
                                pg_loss,
                                entropy_loss,
                                v_loss,
                                clipfrac,
                                approx_kl,
                                ratio,
                                bc_loss,
                                eta,
                            ) = self.model.loss(
                                obs_b,
                                chains_prev_b,
                                chains_next_b,
                                denoising_inds_b,
                                returns_b,
                                values_b,
                                advantages_b,
                                logprobs_b,
                                use_bc_loss=self.use_bc_loss,
                                reward_horizon=self.reward_horizon,
                            )

                            # Total loss calculation
                            loss = (
                                pg_loss
                                + entropy_loss * self.ent_coef
                                + v_loss * self.vf_coef
                                + bc_loss * self.bc_loss_coeff
                            )
                            clipfracs += [clipfrac]

                        # Compute gradients for trainable variables
                        gradients = tape.gradient(loss, self.model.trainable_variables)
                        # Apply gradient clipping if specified
                        if self.max_grad_norm is not None:
                            gradients, _ = tf.clip_by_global_norm(
                                gradients, self.max_grad_norm
                            )
                        # Apply gradients to actor and critic optimizers
                        self.actor_optimizer.apply_gradients(
                            zip(gradients, self.model.trainable_variables)
                        )
                        self.critic_optimizer.apply_gradients(
                            zip(gradients, self.model.trainable_variables)
                        )
                        if self.learn_eta and batch % self.eta_update_interval == 0:
                            eta_gradients = tape.gradient(
                                loss, self.model.eta_variables
                            )
                            if eta_gradients:
                                eta_gradients, _ = tf.clip_by_global_norm(
                                    eta_gradients, self.max_grad_norm
                                )
                                self.eta_optimizer.apply_gradients(
                                    zip(eta_gradients, self.model.eta_variables)
                                )

                        log.info(
                            f"approx_kl: {approx_kl.numpy()}, update_epoch: {update_epoch}, num_batch: {batch}"
                        )

                        # Early stopping if KL divergence exceeds target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            break

                # Calculate explained variance of the value function
                y_pred, y_true = values_k.numpy(), returns_k.numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories if required
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save the model at specified intervals
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    # Logging for evaluation mode
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    # Logging for training mode
                    log.info(f"loss: {type(loss)} {loss.numpy()}")
                    log.info(f"pg_loss: {type(pg_loss)}")
                    log.info(f"v_loss: {type(v_loss)}")
                    log.info(f"bc_loss: {type(bc_loss)}")
                    log.info(f"avg_episode_reward: {type(avg_episode_reward)}")
                    log.info(f"eta: {type(eta)}")
                    log.info(f"t: {type(time)}")
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss.numpy():8.4f} | pg loss {pg_loss.numpy():8.4f} | value loss {v_loss.numpy():8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta.numpy():8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss.numpy(),
                                "value loss": v_loss.numpy(),
                                "bc loss": bc_loss,
                                "eta": eta.numpy(),
                                "approx kl": approx_kl.numpy(),
                                "ratio": ratio.numpy(),
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.learning_rate.numpy(),
                                "critic lr": self.critic_optimizer.learning_rate.numpy(),
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                # Save the run results to a file
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
