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

        # Load dataset
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

        # Convert to TensorFlow tensors
        self.states = tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)
        self.actions = tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)
        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)

        # Make indices
        self.indices = self._make_indices(traj_lengths)

    def _make_indices(self, traj_lengths):
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - self.horizon_steps
            indices.extend([(i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)])
            cur_traj_index += traj_length
        return indices
    
    def __len__(self):
        return len(self.indices)

    @tf.function(reduce_retracing=True)
    def _get_item(self, start, num_before_start):
        end = start + self.horizon_steps
        
        # Get states and actions
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]

        # Stack states
        states_history = []
        for t in range(self.cond_steps-1, -1, -1):
            idx = tf.maximum(num_before_start - t, 0)
            states_history.append(states[idx])
        states = tf.stack(states_history)

        conditions = {"state": states}

        if self.use_img:
            images = self.images[(start - num_before_start):end]
            img_history = []
            for t in range(self.img_cond_steps-1, -1, -1):
                idx = tf.maximum(num_before_start - t, 0)
                img_history.append(images[idx])
            images = tf.stack(img_history)
            conditions["rgb"] = images

        return actions, conditions

    def as_tensorflow_dataset(self, batch_size=32, shuffle=True):
        # Convert indices to tensors
        indices = tf.constant(self.indices, dtype=tf.int32)
        
        # Create dataset from indices
        dataset = tf.data.Dataset.from_tensor_slices(indices)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # Map function to get items
        def map_fn(idx):
            return self._get_item(idx[0], idx[1])

        # Apply mapping and batching
        dataset = dataset.map(
            map_fn, 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            batch_size,
            # drop_remainder=True
        ).prefetch(
            tf.data.AUTOTUNE
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

    def __getitem__(self, idx):
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
        skip last step of truncated episodes
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