"""
No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.
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
    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="/GPU:0",
    ):
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.img_cond_steps = img_cond_steps
        self.use_img = use_img

        # Load dataset with numpy first
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        # Process data
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # Keep as numpy arrays during preprocessing
        self.states_np = dataset["states"][:total_num_steps].astype(np.float32)
        self.actions_np = dataset["actions"][:total_num_steps].astype(np.float32)
        if self.use_img:
            self.images_np = dataset["images"][:total_num_steps].astype(np.float32)

        # Convert indices to numpy arrays and tensors
        self.indices = np.array(self._make_indices(traj_lengths))
        # Make sure to convert to int64 for Dataset.range()
        self.starts = tf.constant(self.indices[:, 0], dtype=tf.int64)
        self.num_before_starts = tf.constant(self.indices[:, 1], dtype=tf.int64)
        # Pre-compute stacked states and images using numpy
        log.info("Pre-computing stacked states...")
        self.cached_states = self._precompute_stacked_states()
        if self.use_img:
            log.info("Pre-computing stacked images...")
            self.cached_images = self._precompute_stacked_images()

        # Convert to tensors after preprocessing
        with tf.device(device):
            self.states = tf.convert_to_tensor(self.states_np)
            self.actions = tf.convert_to_tensor(self.actions_np)
            self.cached_states = tf.convert_to_tensor(self.cached_states)
            if self.use_img:
                self.images = tf.convert_to_tensor(self.images_np)
                self.cached_images = tf.convert_to_tensor(self.cached_images)

        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape: {self.states.shape}")
        log.info(f"Actions shape: {self.actions.shape}")
        if self.use_img:
            log.info(f"Images shape: {self.images.shape}")
    def _make_indices(self, traj_lengths):
        """
        Returns a list of tuples (start_idx, steps_from_start) for each valid sequence.
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - self.horizon_steps
            # For each trajectory, generate all possible starting positions and their offset from trajectory start
            for i in range(cur_traj_index, max_start + 1):
                indices.append((i, i - cur_traj_index))
            cur_traj_index += traj_length
        return indices
    def _precompute_stacked_states(self):
        cached = np.zeros((len(self.indices), self.cond_steps, self.states_np.shape[-1]), dtype=np.float32)
        for i, (start, num_before_start) in enumerate(tqdm(self.indices, desc="Caching states")):
            states = self.states_np[(start - num_before_start):(start + 1)]
            for t in range(self.cond_steps):
                idx = max(num_before_start - t, 0)
                cached[i, self.cond_steps-1-t] = states[idx]
        return cached

    def _precompute_stacked_images(self):
        img_shape = self.images_np.shape[1:]  # (C, H, W) or (H, W, C)
        cached = np.zeros((len(self.indices), self.img_cond_steps, *img_shape), dtype=np.float32)
        for i, (start, num_before_start) in enumerate(tqdm(self.indices, desc="Caching images")):
            images = self.images_np[(start - num_before_start):(start + 1)]
            for t in range(self.img_cond_steps):
                idx = max(num_before_start - t, 0)
                cached[i, self.img_cond_steps-1-t] = images[idx]
        return cached

    @tf.function
    def _get_item_from_index(self, index):
        start = self.starts[index]
        actions = self.actions[start : start + self.horizon_steps]

        conditions = {
            "state": self.cached_states[index],
        }
        if self.use_img:
            conditions["rgb"] = self.cached_images[index]
        return actions, conditions

    def as_tensorflow_dataset(self, batch_size=64, shuffle=True): # Increased batch size
        dataset = tf.data.Dataset.from_tensor_slices(tf.range(len(self.indices), dtype=tf.int64))

        # if shuffle:
        #     dataset = dataset.shuffle(
        #         buffer_size=min(len(self.indices), 10000),
        #         reshuffle_each_iteration=True
        #     )

        # Consider map_and_batch if you have preprocessing in _get_item_from_index
        dataset = dataset.map(
            self._get_item_from_index,
            num_parallel_calls=4,
            deterministic=False
        ).batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=4
        )

        return dataset
    
class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning
    """
    
    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        device="/GPU:0",
        get_mc_return=False,
        **kwargs,
    ):
        """
        Initializes the StitchedSequenceQLearningDataset.

        Args:
            dataset_path (str): Path to the dataset file (.npz or .pkl).
            max_n_episodes (int, optional): Maximum number of episodes to load. Defaults to 10000.
            discount_factor (float, optional): Discount factor for reward-to-go calculation. Defaults to 1.0.
            device (str, optional): Device to use for TensorFlow operations. Defaults to "/GPU:0".
            get_mc_return (bool, optional): Flag to compute Monte Carlo returns. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent class.
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

        with tf.device(device):
            self.rewards = tf.cast(dataset["rewards"][:total_num_steps], tf.float32)
            self.dones = tf.cast(dataset["terminals"][:total_num_steps], tf.float32)

        log.info(f"Rewards shape/dtype: {self.rewards.shape, self.rewards.dtype}")
        log.info(f"Dones shape/dtype: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            **kwargs,
        )
        
        log.info(f"Total number of transitions using: {len(self)}")

        # Compute discounted reward-to-go for each trajectory
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
                    [[i for i in range(prev_traj_length, traj_length)]],
                    returns
                )
                prev_traj_length = traj_length
                
            log.info(f"Computed reward-to-go for each trajectory.")

    @tf.function(reduce_retracing=True)
    def __getitem__(self, idx):
        """
        Retrieves a single transition item.

        Args:
            idx (int): Index of the transition.

        Returns:
            Transition or TransitionWithReturn: The requested transition data.
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start:(start + 1)]
        dones = self.dones[start:(start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps):
                start + 1 + self.horizon_steps
            ]
        else:
            next_states = tf.zeros_like(states)

        # Stack observation history
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

    def make_indices(self, traj_lengths, horizon_steps):
        """
        Creates indices for dataset transitions

        Args:
            traj_lengths (list or array): List of trajectory lengths.
            horizon_steps (int): Number of steps in each horizon.

        Returns:
            list of tuples: List containing (start_index, num_before_start) tuples.
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
            
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices