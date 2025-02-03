
"""
Module for training SAC Diffusion agents using TensorFlow.
"""

from collections import deque
from copy import deepcopy
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

from sklearn.mixture import GaussianMixture

import pickle
import einops
import numpy as np
import tensorflow as tf

from util.scheduler_tf import CosineAnnealingWarmupRestarts
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import logging
import wandb
import math
from tqdm import tqdm

log = logging.getLogger(__name__)
from util.timer import Timer  # Implement or adapt for TensorFlow compatibility
from agent.finetune.train_agent_tf import TrainAgent# Updated base class

def estimate_entropy(actions, num_components=3): # (batch, sample, dim)
    total_entropy = []
    for action in actions:
        gmm = GaussianMixture(n_components=num_components, covariance_type='full')
        gmm.fit(action)
        weights = gmm.weights_
        entropies = []
        for i in range(gmm.n_components):
            cov_matrix = gmm.covariances_[i]
            d = cov_matrix.shape[0]
            entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
            entropies.append(entropy)
        entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
        total_entropy.append(entropy)
    final_entropy = sum(total_entropy) / len(total_entropy)
    return final_entropy

class TrainSACDiffusionAgent(TrainAgent):
    """
    Trainer for SAC Diffusion agents using TensorFlow.

    Args:
        cfg: Configuration object containing training parameters and settings.
    """

    def __init__(self, cfg):
        """
        Initializes the TrainSACDiffusionAgent with the given configuration.

        Args:
            cfg: Configuration object containing training parameters and settings.
        """
        super().__init__(cfg)

        self.gamma = cfg.train.gamma
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr
        self.buffer_size = cfg.train.buffer_size
        self.update_per_iteration = cfg.train.update_per_iteration
        self.delay_alpha_update = cfg.train.delay_alpha_update
        self.num_sample = cfg.train.num_sample
        self.delay_update = cfg.train.delay_update
        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor
        self.batch_size = cfg.train.batch_size

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
            weight_decay=cfg.train.actor_weight_decay,
            epsilon=1e-8  # TensorFlow uses decay differently
        )

        self.q1_lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.q1_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.q1_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
            epsilon=1e-8  # TensorFlow uses decay differently
        )

        self.q2_lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.q2_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.q2_lr_scheduler,
            weight_decay=cfg.train.critic_weight_decay,
            epsilon=1e-8  # TensorFlow uses decay differently
        )

        self.alpha_lr_scheduler = CosineAnnealingWarmupRestarts(
            first_cycle_steps=cfg.train.alpha_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.alpha_lr,
            min_lr=cfg.train.alpha_lr_scheduler.min_lr,
            warmup_steps=cfg.train.alpha_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.alpha_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.alpha_lr_scheduler,
            weight_decay=cfg.train.alpha_weight_decay,
            epsilon=1e-8  # TensorFlow uses decay differently
        )
        
        

    def run(self):
        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        log_alpha_loss = tf.constant(0.0, dtype=tf.float32)

        # Start training loop
        timer = Timer()
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
            
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            prev_obs_tensor = tf.convert_to_tensor(prev_obs_venv["state"], dtype=tf.float32)

            with tf.device(self.device):
                for step in range(self.n_steps):
                    if step % 10 == 0:
                        print(f"Processed step {step} of {self.n_steps}")
                    
                    cond = {
                        "state": prev_obs_tensor
                    }

                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        training=False
                    )

                    action_venv = samples.trajectories[:, : self.act_steps].numpy()
                    # action_venv shape (n_env, act_steps, act_dim)

                    # Apply selected actions to the environments
                    (
                        obs_venv,
                        reward_venv,
                        terminated_venv,
                        truncated_venv,
                        info_venv,
                    ) = self.venv.step(action_venv)


                    done_venv = terminated_venv | truncated_venv
                    reward_trajs[step] = reward_venv
                    firsts_trajs[step + 1] = done_venv

                    # add to buffer
                    if not eval_mode:
                        for i in range(self.n_envs):
                            obs_buffer.append(prev_obs_venv["state"][i])
                            # print(obs_buffer[0])
                            if truncated_venv[i]:
                                next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                            else:
                                next_obs_buffer.append(obs_venv["state"][i])
                            action_buffer.append(action_venv[i])
                        reward_buffer.extend(reward_venv * self.scale_reward_factor)
                        terminated_buffer.extend(terminated_venv)

                    prev_obs_tensor = tf.convert_to_tensor(obs_venv["state"], dtype=tf.float32)
                    
                    cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

                    
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
                obs_array = np.array(obs_buffer)
                next_obs_array = np.array(next_obs_buffer)
                action_array = np.array(action_buffer)
                reward_array = np.array(reward_buffer)
                terminated_array = np.array(terminated_buffer)

                for update in range(self.update_per_iteration):
                    
                    inds = np.random.choice(len(obs_array), self.batch_size)

                    obs_b = tf.convert_to_tensor(obs_array[inds], dtype=tf.float32)
                    next_obs_b = tf.convert_to_tensor(next_obs_array[inds], dtype=tf.float32)
                    action_b = tf.convert_to_tensor(action_array[inds], dtype=tf.float32)
                    reward_b = tf.convert_to_tensor(reward_array[inds], dtype=tf.float32)
                    done_b = tf.convert_to_tensor(terminated_array[inds], dtype=tf.float32)


                    obs_b = {"state": obs_b}
                    next_obs_b = {"state": next_obs_b}

                    # compute target q
                    next_action = self.model(next_obs_b, training=False)
                    next_q1_target = self.model.q1_target(next_obs_b, next_action.trajectories, training=False)
                    next_q2_target = self.model.q2_target(next_obs_b, next_action.trajectories, training=False)
                    
                    min_nest_q_target = tf.minimum(next_q1_target, next_q2_target)

                    next_q_value = reward_b + (1 - done_b) * self.gamma * min_nest_q_target

                    ## -- core --##
                    with tf.GradientTape() as tape:
                        q1_a_value = self.model.q1(obs_b, action_b, training=True)
                        q2_a_value = self.model.q2(obs_b, action_b, training=True)

                        q1_loss = tf.reduce_mean((q1_a_value - next_q_value) ** 2)
                        q2_loss = tf.reduce_mean((q2_a_value - next_q_value) ** 2)
                        # print("Watched variables:", tape.watched_variables())
                        q_loss = q1_loss + q2_loss

                    q_var = self.model.q1.trainable_variables + self.model.q2.trainable_variables
                    q_gradients = tape.gradient(q_loss, q_var)

                    q1_gradients = q_gradients[:len(self.model.q1.trainable_variables)]
                    q2_gradients = q_gradients[len(self.model.q1.trainable_variables):len(self.model.q1.trainable_variables) + len(self.model.q2.trainable_variables)]

                    self.q1_optimizer.apply_gradients(zip(q1_gradients, self.model.q1.trainable_variables))
                    self.q2_optimizer.apply_gradients(zip(q2_gradients, self.model.q2.trainable_variables))

                    # update entropy
                    if self.itr % self.delay_alpha_update == 0 and self.model.learn_alpha:
                        actions = []
                        for _ in range(self.num_sample):
                            actions.append(self.model(obs_b, training=False).trajectories[:,0,:])
                        actions = tf.stack(actions, axis=0)
                        actions = einops.rearrange(actions, 's b a -> b s a')
                        entropy = estimate_entropy(actions)
                        log.info(f"Updated Emtropy: {entropy}")
                        self.model.entropy = tf.constant(entropy, dtype=tf.float32)
                    
                    # optimization for policy
                    if self.itr % self.delay_update == 0:
                        with tf.GradientTape() as tape:
                            pi = self.model(obs_b)
                            q1_pi = self.model.q1(obs_b, pi.trajectories)
                            q2_pi = self.model.q2(obs_b, pi.trajectories)
                            min_q_pi = tf.minimum(q1_pi, q2_pi)
                            actor_loss = tf.reduce_mean(-min_q_pi)
    
                        actor_var = self.model.actor_ft.trainable_variables
                        actor_gradients = tape.gradient(actor_loss, actor_var)
                        self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_var))

                    # update alpha
                    if self.itr % self.delay_alpha_update == 0 and self.model.learn_alpha:
                        with tf.GradientTape() as tape:
                          log_alpha_loss = -tf.reduce_mean(self.model.logalpha * (-self.model.entropy + self.model.target_entropy))
                        alpha_var = self.model.logalpha
                        alpha_gradients = tape.gradient(log_alpha_loss, alpha_var)
                        self.alpha_optimizer.apply_gradients([(alpha_gradients, alpha_var)])
                    
                    if self.itr % self.delay_update == 0:
                        self.model.update_target()

            if self.itr >= self.n_critic_warmup_itr:
                itr = max(self.itr - self.n_critic_warmup_itr, 0)
                actor_lr = self.actor_lr_scheduler(itr)
                self.actor_optimizer.learning_rate.assign(actor_lr)

            q1_lr = self.q1_lr_scheduler(self.itr)
            q2_lr = self.q2_lr_scheduler(self.itr)
            alpha_lr = self.alpha_lr_scheduler(self.itr)

            self.q1_optimizer.learning_rate.assign(q1_lr)
            self.q2_optimizer.learning_rate.assign(q2_lr)
            self.alpha_optimizer.learning_rate.assign(alpha_lr)

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )

            # Save the model at specified intervals
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

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
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | q1_loss: {q1_loss:8.4f} | q2_loss: {q2_loss:8.4f} | q_loss: {q_loss:8.4f} | actor loss: {actor_loss:8.4f} | entropy: {self.model.entropy:8.4f} | log alpha loss: {log_alpha_loss:8.4f} | alpha: {self.model.logalpha.numpy():8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "q1_loss": q1_loss,
                                "q2_loss": q2_loss,
                                "q_loss": q_loss,
                                "actor loss": actor_loss,
                                "entropy": self.model.entropy,
                                "log alpha loss": log_alpha_loss,
                                "alpha": self.model.logalpha,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "actor lr": self.actor_optimizer.learning_rate.numpy(),
                                "q1 lr": self.q1_optimizer.learning_rate.numpy(),
                                "q2 lr": self.q2_optimizer.learning_rate.numpy(),
                                "alpha lr": self.alpha_optimizer.learning_rate.numpy(),
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                # Save the run results to a file
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            
            # log loss and save metrics
            self.itr += 1
