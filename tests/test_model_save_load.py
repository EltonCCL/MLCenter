import pytest
import os
import tempfile
from agent.pretrain.train_diffusion_agent_tf import TrainDiffusionAgent
from omegaconf import OmegaConf
import math
import numpy as np
from copy import deepcopy
import tensorflow as tf

# Configure GPU at module level
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def test_save_and_load_model():
    # Load configuration
    cfg = OmegaConf.load('/home/elton/dppo/cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp_tf.yaml')
    cfg.train_dataset_path = "tests/dummy_data/train_dummy.npz"
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        cfg.logdir = tmpdirname
        agent = TrainDiffusionAgent(cfg)
        agent.save_model()
        agent.load(agent.epoch)
        assert os.path.exists(os.path.join(tmpdirname, "checkpoint")), "Checkpoint directory should exist"

def test_train_save_load_compare():
    # Load configuration
    cfg = OmegaConf.load('/home/elton/dppo/cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp_tf.yaml')
    cfg.train_dataset_path = "tests/dummy_data/train_dummy.npz"
    with tempfile.TemporaryDirectory() as tmpdirname:
        cfg.logdir = tmpdirname
        agent = TrainDiffusionAgent(cfg)
        
        # Get initial weights before any training
        initial_weights = deepcopy(agent.model.get_weights())
        
        # Train the model for a limited number of epochs
        agent.n_epochs = 1
        agent.run()
        
        # Save the model
        agent.save_model()
        
        # Capture model weights after training
        trained_weights = deepcopy(agent.model.get_weights())
        
        # Verify that training actually changed the weights
        weights_changed = False
        for init, trained in zip(initial_weights, trained_weights):
            if not np.array_equal(init, trained):
                weights_changed = True
                break
        assert weights_changed, "Model weights should change after training"
        
        # Load the saved model
        agent.load(agent.epoch)
        
        # Capture model weights after loading
        loaded_weights = deepcopy(agent.model.get_weights())
        
        # Compare the trained and loaded model weights
        for trained, loaded in zip(trained_weights, loaded_weights):
            assert np.allclose(trained, loaded, atol=1e-6), "Model weights should match after loading"


if __name__ == "__main__":
    pytest.main([__file__])