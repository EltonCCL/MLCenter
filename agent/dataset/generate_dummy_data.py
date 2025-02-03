"""
This module generates dummy data for training purposes. It creates random trajectories
with specified numbers of episodes and steps, and saves the generated states and actions
to a NumPy .npz file.
"""

import numpy as np
import pickle

def generate_dummy_data(num_episodes=100, max_steps=500, use_img=False):
    traj_lengths = np.random.randint(20, 50, size=num_episodes) 
    total_num_steps = np.sum(traj_lengths)
    
    states = np.random.randn(total_num_steps, 11).astype(np.float32)
    actions = np.random.randn(total_num_steps, 3).astype(np.float32)
    
    data = {
        "traj_lengths": traj_lengths,
        "states": states,
        "actions": actions
    }
    
    np.savez("train_dummy.npz", **data)
    print("Dummy data generated and saved to train_dummy.npz")

if __name__ == "__main__":
    generate_dummy_data()