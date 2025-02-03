import tensorflow as tf
import torch
import numpy as np
from typing import Tuple
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both versions (assuming they're in separate files)
# PyTorch version
from model.diffusion.sampling import cosine_beta_schedule as torch_cosine_beta_schedule
from model.diffusion.sampling import extract as torch_extract
from model.diffusion.sampling import make_timesteps as torch_make_timesteps

# TensorFlow version
from model.diffusion.sampling_tf import cosine_beta_schedule as tf_cosine_beta_schedule
from model.diffusion.sampling_tf import extract as tf_extract
from model.diffusion.sampling_tf import make_timesteps as tf_make_timesteps

@pytest.fixture(scope="module")
def device_setup():
    """Setup devices for both frameworks"""
    # torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tf_device = 'GPU:0' if torch.cuda.is_available() else 'CPU:0'
    torch_device = torch.device('cpu')
    tf_device = '/CPU:0'
    return torch_device, tf_device

@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set random seeds for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

def assert_close(torch_tensor: torch.Tensor, tf_tensor: tf.Tensor, 
                rtol: float = 1e-5, atol: float = 1e-8):
    """Helper function to compare PyTorch and TensorFlow tensors"""
    torch_np = torch_tensor.cpu().numpy()
    tf_np = tf_tensor.numpy()
    np.testing.assert_allclose(torch_np, tf_np, rtol=rtol, atol=atol)

@pytest.mark.parametrize("timesteps,s", [
    (10, 0.008),
    (100, 0.1),
    (1000, 0.5),
    (1, 0.008),  # edge case
])

def test_cosine_beta_schedule(timesteps, s):
    """Test cosine beta schedule implementation"""
    torch_betas = torch_cosine_beta_schedule(timesteps, s)
    tf_betas = tf_cosine_beta_schedule(timesteps, s)
    
    assert_close(torch_betas, tf_betas)
    assert torch_betas.shape == tf_betas.shape
    assert torch_betas.dtype == torch.float32
    assert tf_betas.dtype == tf.float32

@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 10),
    (4, 100),
    (16, 10),
    (1, 1),  # edge case
])
def test_extract(device_setup, batch_size, seq_len):
    """Test extract function implementation"""
    torch_device, tf_device = device_setup
    
    # Create test data
    a_np = np.random.randn(seq_len)
    t_np = np.random.randint(0, seq_len, size=(batch_size,))
    x_shape = (batch_size, 4, 4)  # Example shape

    # Convert to respective formats
    a_torch = torch.from_numpy(a_np).to(torch_device)
    t_torch = torch.from_numpy(t_np).to(torch_device)
    
    a_tf = tf.convert_to_tensor(a_np)
    t_tf = tf.convert_to_tensor(t_np)

    # Run both implementations
    torch_result = torch_extract(a_torch, t_torch, x_shape)
    tf_result = tf_extract(a_tf, t_tf, x_shape)

    assert_close(torch_result, tf_result)

@pytest.mark.parametrize("batch_size,timestep", [
    (1, 0),
    (4, 10),
    (16, 100),
    (1, 0),  # edge case
])
def test_make_timesteps(device_setup, batch_size, timestep):
    """Test make_timesteps function implementation"""
    torch_device, tf_device = device_setup
    
    torch_result = torch_make_timesteps(batch_size, timestep, torch_device)
    tf_result = tf_make_timesteps(batch_size, timestep, tf_device)

    assert_close(torch_result, tf_result)
    assert torch_result.shape == tf_result.shape
    assert torch.all(torch_result == timestep)
    assert tf.reduce_all(tf_result == timestep)

def test_numerical_stability(device_setup):
    """Test numerical stability with extreme values"""
    torch_device, tf_device = device_setup
    
    # Test with very large timesteps
    large_timesteps = 10000
    torch_betas = torch_cosine_beta_schedule(large_timesteps)
    tf_betas = tf_cosine_beta_schedule(large_timesteps)
    assert_close(torch_betas, tf_betas)

    # Test with very small s value
    small_s = 1e-10
    torch_betas = torch_cosine_beta_schedule(100, small_s)
    tf_betas = tf_cosine_beta_schedule(100, small_s)
    assert_close(torch_betas, tf_betas)

def test_device_consistency(device_setup):
    """Test device placement consistency"""
    torch_device, tf_device = device_setup
    
    # Test device placement for make_timesteps
    batch_size = 4
    timestep = 10
    
    torch_result = torch_make_timesteps(batch_size, timestep, torch_device)
    tf_result = tf_make_timesteps(batch_size, timestep, tf_device)
    
    # Check if tensor is on the correct device (PyTorch)
    assert torch_result.device.type == torch_device.type
    
    # For TensorFlow, we can check if the device string contains the expected device
    if tf_device.startswith('GPU'):
        assert tf_result.device.endswith('GPU:0')
    else:
        assert tf_result.device.endswith('CPU:0')

def test_dtype_consistency():
    """Test dtype consistency across implementations"""
    timesteps = 100
    
    # Test float32 (default)
    torch_betas = torch_cosine_beta_schedule(timesteps)
    tf_betas = tf_cosine_beta_schedule(timesteps)
    assert torch_betas.dtype == torch.float32
    assert tf_betas.dtype == tf.float32
    
    # Test float64
    torch_betas = torch_cosine_beta_schedule(timesteps, dtype=torch.float64)
    tf_betas = tf_cosine_beta_schedule(timesteps, dtype=tf.float64)
    assert torch_betas.dtype == torch.float64
    assert tf_betas.dtype == tf.float64

if __name__ == "__main__":
    pytest.main(['tests/test_sampling.py'])