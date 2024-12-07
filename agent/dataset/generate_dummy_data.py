import numpy as np
import pickle

def generate_dummy_data(num_episodes=100, max_steps=500, use_img=False):
    traj_lengths = np.random.randint(400, 500, size=num_episodes)  # Around 470
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