"""
TrainAgent module for fine-tuning agents using TensorFlow.
"""

import os
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
import random
import hydra
import logging
import wandb

log = logging.getLogger(__name__)
from env.gym_utils import make_async  # Ensure this is compatible with TensorFlow


class TrainAgent:
    """
    Handles the training process of agents using TensorFlow.

    Args:
        cfg: Configuration object containing training parameters and settings.
    """

    def __init__(self, cfg):
        """
        Initializes the TrainAgent with the given configuration.

        Args:
            cfg: Configuration object containing training parameters and settings.
        """
        super().__init__()
        self.cfg = cfg
        self.device = (
            cfg.device
        )  # In TensorFlow, device placement is handled differently
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.get("wandb", None)
        if self.use_wandb is not None:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )  # Ensure different seeds for each env
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.save_full_observations = cfg.env.get("save_full_observations", False)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        # Ensure the TensorFlow model is compiled if necessary

        # Training params
        self.itr = 0
        self.n_train_itr = cfg.train.n_train_itr
        self.val_freq = cfg.train.val_freq
        self.force_train = cfg.train.get("force_train", False)
        self.n_steps = cfg.train.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_trajs = cfg.train.get("save_trajs", False)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.render_freq = cfg.train.render.freq
        self.n_render = cfg.train.render.num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        self.traj_plotter = (
            hydra.utils.instantiate(cfg.train.plotter)
            if "plotter" in cfg.train
            else None
        )


    def run(self):
        """
        Executes the training loop.
        """
        pass

    def save_model(self):
        """
        Saves the current model checkpoint to disk.
        """
        checkpoint = tf.train.Checkpoint(
            itr=tf.Variable(self.itr),
            ft_model=self.model,
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            max_to_keep=100
        )
        save_path = manager.save(checkpoint_number=self.itr)
        log.info(f"Saved model to {save_path}")

    def load(self, itr):
        assert False, "Not implemented"
        # """
        # Loads a model checkpoint from disk.

        # Args:
        #     itr: The iteration number of the checkpoint to load.
        # """
        # loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.ckpt")
        # self.checkpoint.restore(loadpath).assert_consumed()
        # self.itr = int(self.checkpoint.itr.numpy())
        # log.info(f"Loaded model from {loadpath}")

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        """
        Resets all environments with the given options.

        Args:
            verbose (bool): If True, logs the reset actions.
            options_venv: Specific options for each environment.
            **kwargs: Additional keyword arguments for environment reset.

        Returns:
            obs_venv: The observations after resetting the environments.
        """
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # Convert to dictionary of stacked numpy arrays if necessary
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        """
        Resets a single environment specified by its index.

        Args:
            env_ind: The index of the environment to reset.
            verbose (bool): If True, logs the reset action.

        Returns:
            obs: The observation after resetting the environment.
        """
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
