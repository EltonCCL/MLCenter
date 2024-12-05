"""
Pre-training data loader converted to TensorFlow.
Original PyTorch version modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py
"""

from collections import namedtuple
import numpy as np
import tensorflow as tf
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)

class StitchedSequenceDataset:
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.
    Note: Changed to not inherit from tf.data.Dataset for simpler implementation.
    """
    
    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
    ):
        """
        Initialize the stitched sequence dataset.
        
        Args:
            dataset_path (str): Path to the dataset file.
            horizon_steps (int): Number of steps in the horizon.
            cond_steps (int): Number of conditional steps.
            img_cond_steps (int): Number of image conditional steps.
            max_n_episodes (int): Maximum number of episodes to load.
            use_img (bool): Whether to use image data.
        """
        assert img_cond_steps <= cond_steps, "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
            
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Convert numpy arrays to TF tensors
        self.states = tf.convert_to_tensor(
            dataset["states"][:total_num_steps], dtype=tf.float32
        )
        self.actions = tf.convert_to_tensor(
            dataset["actions"][:total_num_steps], dtype=tf.float32
        )
        
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/dtype: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/dtype: {self.actions.shape, self.actions.dtype}")
        
        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps])
            log.info(f"Images shape/dtype: {self.images.shape, self.images.dtype}")

    def __getitem__(self, idx):
        """
        Retrieve a batch at the specified index.
        
        Args:
            idx (int): Index of the batch.
        
        Returns:
            Batch: A namedtuple containing actions and conditions.
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        
        states_list = [
            states[max(num_before_start - t, 0)]
            for t in reversed(range(self.cond_steps))
        ]
        states = tf.stack(states_list)
        
        conditions = {"state": states}
        
        if self.use_img:
            images = self.images[(start - num_before_start):end]
            images_list = [
                images[max(num_before_start - t, 0)]
                for t in reversed(range(self.img_cond_steps))
            ]
            images = tf.stack(images_list)
            conditions["rgb"] = images
            
        return Batch(actions, conditions)

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Create indices for sampling batches.
        
        Args:
            traj_lengths (list): List of trajectory lengths.
            horizon_steps (int): Number of steps in the horizon.
        
        Returns:
            list: List of tuples containing start index and offset.
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def __len__(self):
        """
        Get the total number of batches.
        
        Returns:
            int: Number of batches.
        """
        return len(self.indices)

    def to_tf_dataset(self):
        """
        Convert the stitched sequence dataset to a tf.data.Dataset.
        
        Returns:
            tf.data.Dataset: TensorFlow dataset.
        """
        return tf.data.Dataset.from_tensor_slices(
            range(len(self))
        ).map(lambda idx: tf.py_function(
            self.__getitem__,
            [idx],
            [tf.float32, {
                'state': tf.float32,
                'rgb': tf.float32 if self.use_img else None
            }]
        ))


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Stitched sequence dataset tailored for Q-Learning, including reward-to-go computations.
    """
    
    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        get_mc_return=False,
        **kwargs,
    ):
        """
        Initialize the Q-Learning stitched sequence dataset.
        
        Args:
            dataset_path (str): Path to the dataset file.
            max_n_episodes (int): Maximum number of episodes to load.
            discount_factor (float): Discount factor for future rewards.
            get_mc_return (bool): Whether to compute Monte Carlo returns.
            **kwargs: Additional arguments for the parent class.
        """
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
            
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        self.discount_factor = discount_factor
        
        # Convert rewards and dones to TF tensors
        self.rewards = tf.convert_to_tensor(
            dataset["rewards"][:total_num_steps], dtype=tf.float32
        )
        self.dones = tf.cast(
            tf.convert_to_tensor(dataset["terminals"][:total_num_steps]),
            dtype=tf.float32
        )
        
        log.info(f"Rewards shape/dtype: {self.rewards.shape, self.rewards.dtype}")
        log.info(f"Dones shape/dtype: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            **kwargs,
        )
        
        log.info(f"Total number of transitions using: {len(self)}")

        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = tf.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = tf.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns = tf.tensor_scatter_nd_update(
                        returns,
                        [[-(t + 1)]],
                        [traj_rewards[-(t + 1)] + self.discount_factor * prev_return]
                    )
                    prev_return = returns[-(t + 1)]
                self.reward_to_go = tf.tensor_scatter_nd_update(
                    self.reward_to_go,
                    [[i] for i in range(prev_traj_length, traj_length)],
                    returns
                )
                prev_traj_length = traj_length
                
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Create indices for sampling batches, accounting for done flags.
        
        Args:
            traj_lengths (list): List of trajectory lengths.
            horizon_steps (int): Number of steps in the horizon.
        
        Returns:
            list: List of tuples containing start index and offset.
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        """
        Retrieve a transition at the specified index.
        
        Args:
            idx (int): Index of the transition.
        
        Returns:
            Transition or TransitionWithReturn: A namedtuple containing transition data.
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start:(start + 1)]
        dones = self.dones[start:(start + 1)]

        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps):
                start + 1 + self.horizon_steps
            ]
        else:
            next_states = tf.zeros_like(states)

        states = tf.stack([
            states[max(num_before_start - t, 0)]
            for t in reversed(range(self.cond_steps))
        ])
        next_states = tf.stack([
            next_states[max(num_before_start - t, 0)]
            for t in reversed(range(self.cond_steps))
        ])
        
        conditions = {"state": states, "next_state": next_states}
        
        if self.use_img:
            images = self.images[(start - num_before_start):end]
            images = tf.stack([
                images[max(num_before_start - t, 0)]
                for t in reversed(range(self.img_cond_steps))
            ])
            conditions["rgb"] = images
            
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start:(start + 1)]
            return TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            return Transition(
                actions,
                conditions,
                rewards,
                dones,
            )