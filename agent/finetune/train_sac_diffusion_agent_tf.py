
"""
Module for training SAC Diffusion agents using TensorFlow.
"""

from collections import deque
import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

import pickle
import einops
import numpy as np
import tensorflow as tf
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
        """
        q1 QNet
        q2 QNet
        target_q1 QNet
        target_q2 QNet
        policy DACERPolicyNet
        log_alpha tf.Variable default math.log(3)
        actor_optimizer Adam
        policy_optimizer Adam
        alpha_optimizer Adam
        buffer ReplayBuffer
        """

    def run(self):
        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

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
                        obs_buffer.append(prev_obs_venv["state"])
                        action_buffer.append(action_venv)
                        reward_buffer.append(reward_venv * self.scale_reward_factor)
                        terminated_buffer.append(terminated_venv)

                    prev_obs_tensor = tf.convert_to_tensor(obs_venv["state"], dtype=tf.float32)

                    exit()
                # Update the replay buffer
            self.itr += 1