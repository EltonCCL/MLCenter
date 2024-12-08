# tests/test_diffusion_comparison.py

import pytest
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from model.diffusion.diffusion import DiffusionModel as PyTorchDiffusionModel
from model.diffusion.diffusion_tf import DiffusionModel as TensorFlowDiffusionModel, Sample as TF_Sample
from collections import namedtuple

@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and TensorFlow.
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

# Mock network classes shortened for brevity - same as your code
class MockNetworkPyTorch(nn.Module):
    """
    Mock PyTorch network that returns a tensor of 0.5s.
    """
    def forward(self, x, t, cond=None):
        return torch.ones_like(x) * 0.5

class MockNetworkTensorFlow(tf.keras.Model):
    """
    Mock TensorFlow network that returns a tensor of 0.5s.
    """
    def call(self, x, t, cond=None):
        return tf.ones_like(x) * 0.5


@pytest.fixture(scope="module")
def model_configs():
    """
    Fixture to initialize the PyTorch Diffusion Model with a mock network.
    """
    return {
        "horizon_steps": 1,
        "obs_dim": 5,
        "action_dim": 3,
        "denoising_steps": 3,
        "predict_epsilon": True,
        "use_ddim": False
    }

@pytest.fixture(scope="module")
def torch_model(model_configs):
    network = MockNetworkPyTorch()
    return PyTorchDiffusionModel(
        network=network,
        device="cpu",
        **model_configs
    )

@pytest.fixture(scope="module")
def tf_model(model_configs):
    network = MockNetworkTensorFlow()
    return TensorFlowDiffusionModel(
        network=network,
        device="/CPU:0",
        **model_configs
    )

@pytest.fixture(scope="module")
def sample_inputs(model_configs):
    """Generate sample inputs for testing"""
    batch_size = 2
    fixed_noise = np.random.randn(
        batch_size, 
        model_configs["horizon_steps"], 
        model_configs["action_dim"]
    ).astype(np.float32)
    
    cond_state = np.random.rand(
        batch_size, 
        model_configs["horizon_steps"], 
        model_configs["obs_dim"]
    ).astype(np.float32)
    
    return {
        "fixed_noise": fixed_noise,
        "cond_state": cond_state
    }

def test_noise_schedule_parameters(torch_model, tf_model):
    """Test if noise schedule parameters match between PyTorch and TensorFlow models"""
    torch_alphas = torch_model.alphas_cumprod.detach().numpy()
    tf_alphas = tf_model.alphas_cumprod.numpy()
    
    assert np.allclose(torch_alphas, tf_alphas, atol=1e-6), \
        "Alphas cumprod values don't match between implementations"
    
    torch_betas = torch_model.betas.detach().numpy()
    tf_betas = tf_model.betas.numpy()
    
    assert np.allclose(torch_betas, tf_betas, atol=1e-6), \
        "Beta values don't match between implementations"

def test_forward_pass(torch_model, tf_model, sample_inputs):
    """Test if forward pass outputs match between PyTorch and TensorFlow models"""
    # Prepare inputs for both frameworks
    torch_noise = torch.tensor(sample_inputs["fixed_noise"])
    torch_cond = {"state": torch.tensor(sample_inputs["cond_state"])}
    
    tf_noise = tf.constant(sample_inputs["fixed_noise"])
    tf_cond = {"state": tf.constant(sample_inputs["cond_state"])}
    
    # Get model outputs
    torch_output = torch_model(
        torch_cond, 
        deterministic=True, 
        fixed_noise=torch_noise
    ).trajectories.detach().numpy()
    
    tf_output = tf_model(
        tf_cond, 
        deterministic=True, 
        fixed_noise=tf_noise
    ).trajectories.numpy()
    
    # Compare outputs
    assert np.allclose(torch_output, tf_output, atol=1e-4), \
        "Model outputs don't match between implementations"

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(torch_model, tf_model, model_configs, batch_size):
    """Test model behavior with different batch sizes"""
    # Generate inputs for the specific batch size
    noise = np.random.randn(
        batch_size, 
        model_configs["horizon_steps"], 
        model_configs["action_dim"]
    ).astype(np.float32)
    
    cond = np.random.rand(
        batch_size, 
        model_configs["horizon_steps"], 
        model_configs["obs_dim"]
    ).astype(np.float32)
    
    # Convert to framework-specific formats
    torch_output = torch_model(
        {"state": torch.tensor(cond)},
        deterministic=True,
        fixed_noise=torch.tensor(noise)
    ).trajectories.detach().numpy()
    
    tf_output = tf_model(
        {"state": tf.constant(cond)},
        deterministic=True,
        fixed_noise=tf.constant(noise)
    ).trajectories.numpy()
    
    assert np.allclose(torch_output, tf_output, atol=1e-4), \
        f"Outputs don't match for batch size {batch_size}"
if __name__ == "__main__":
    pytest.main(['tests/test_diffusion_combine.py'])