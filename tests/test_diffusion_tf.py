"""
Module containing tests for the DiffusionModel using TensorFlow.
"""

import pytest
import tensorflow as tf
import numpy as np
from model.diffusion.diffusion_tf import DiffusionModel, Sample
from collections import namedtuple

# Create a simple mock network for testing
class MockNetwork(tf.keras.Model):
    """Mock TensorFlow network that returns a tensor of ones with the same shape as input."""
    def call(self, x, t, cond=None):
        # Return a tensor of ones with the same shape as input
        return tf.ones_like(x)

# Fix random seeds to ensure test consistency
@pytest.fixture(scope="session", autouse=True)
def set_random_seeds():
    """Fixture to set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

@pytest.fixture(scope="module")
def mock_network():
    """Fixture to initialize the MockNetwork."""
    return MockNetwork()

@pytest.fixture(scope="module")
def diffusion_model(mock_network):
    """Fixture to initialize the DiffusionModel with default parameters."""
    horizon_steps = 10
    obs_dim = 5
    action_dim = 3
    denoising_steps = 50
    device = "/CPU:0"  # Use CPU for testing

    model = DiffusionModel(
        network=mock_network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        predict_epsilon=True,
        use_ddim=False
    )
    return model

@pytest.fixture(scope="module")
def diffusion_model_ddim(mock_network):
    """Fixture to initialize the DiffusionModel with DDIM configuration."""
    horizon_steps = 10
    obs_dim = 5
    action_dim = 3
    denoising_steps = 50
    device = "/CPU:0"  # Use CPU for testing
    ddim_steps = 10

    model = DiffusionModel(
        network=mock_network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        predict_epsilon=True,
        use_ddim=True,
        ddim_discretize="uniform",
        ddim_steps=ddim_steps
    )
    return model

def test_initialization(diffusion_model):
    """Test the initialization of the DiffusionModel."""
    # Check if the model is correctly initialized
    assert isinstance(diffusion_model, tf.keras.Model)
    assert diffusion_model.horizon_steps == 10
    assert diffusion_model.obs_dim == 5
    assert diffusion_model.action_dim == 3
    assert diffusion_model.denoising_steps == 50
    assert diffusion_model.device == "/CPU:0"
    assert diffusion_model.predict_epsilon is True
    assert diffusion_model.use_ddim is False

    # Check if key attributes are properly set
    assert diffusion_model.betas is not None
    assert diffusion_model.alphas is not None
    assert diffusion_model.alphas_cumprod is not None
    assert diffusion_model.ddpm_var is not None

def test_forward_pass(diffusion_model):
    """Test the forward pass of the DiffusionModel."""
    # Create mock conditional input
    batch_size = 4
    sample_data = tf.random.uniform((batch_size, diffusion_model.horizon_steps, diffusion_model.obs_dim), dtype=tf.float32)
    cond = {"state": sample_data}

    # Perform forward pass
    sample = diffusion_model(cond)

    # Check return type
    assert isinstance(sample, Sample)
    
    # Check output shape
    assert sample.trajectories.shape == (batch_size, diffusion_model.horizon_steps, diffusion_model.action_dim)
    assert sample.chains is None

def test_loss_computation(diffusion_model):
    """Test the loss computation of the DiffusionModel."""
    # Create mock input data
    batch_size = 4
    x_start = tf.random.uniform((batch_size, diffusion_model.horizon_steps, diffusion_model.action_dim), dtype=tf.float32)
    cond = {"state": tf.random.uniform((batch_size, diffusion_model.horizon_steps, diffusion_model.obs_dim), dtype=tf.float32)}

    # Compute loss
    loss = diffusion_model.loss(x_start, cond)

    # Check loss type and shape
    assert isinstance(loss, tf.Tensor)
    assert loss.shape == ()
    assert loss.numpy() >= 0.0

def test_p_losses_epsilon(diffusion_model):
    """Test the p_losses computation when predicting epsilon."""
    # Create mock input data
    batch_size = 4
    x_start = tf.random.uniform((batch_size, diffusion_model.horizon_steps, diffusion_model.action_dim), dtype=tf.float32)
    cond = {"state": tf.random.uniform((batch_size, diffusion_model.horizon_steps, diffusion_model.obs_dim), dtype=tf.float32)}
    t = tf.random.uniform([batch_size], 0, diffusion_model.denoising_steps, dtype=tf.int32)

    # Compute p_losses
    losses = diffusion_model.p_losses(x_start, cond, t)

    # Check loss type and shape
    assert isinstance(losses, tf.Tensor)
    assert losses.shape == ()
    assert losses.numpy() >= 0.0

def test_p_losses_x0_prediction():
    """Test the p_losses computation when predicting x0."""
    horizon_steps = 10
    obs_dim = 5
    action_dim = 3
    denoising_steps = 50
    device = "/CPU:0"  # Use CPU for testing

    # Initialize model to directly predict x0
    mock_network_instance = MockNetwork()
    model = DiffusionModel(
        network=mock_network_instance,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        predict_epsilon=False,
        use_ddim=False
    )

    # Create mock input data
    batch_size = 4
    x_start = tf.random.uniform((batch_size, horizon_steps, action_dim), dtype=tf.float32)
    cond = {"state": tf.random.uniform((batch_size, horizon_steps, obs_dim), dtype=tf.float32)}
    t = tf.random.uniform([batch_size], 0, denoising_steps, dtype=tf.int32)

    # Compute p_losses
    losses = model.p_losses(x_start, cond, t)

    # Check loss type and shape
    assert isinstance(losses, tf.Tensor)
    assert losses.shape == ()
    assert losses.numpy() >= 0.0

def test_sample_shape_ddim(diffusion_model_ddim):
    """Test the sample shape of the DiffusionModel with DDIM."""
    # Create mock conditional input
    batch_size = 4
    sample_data = tf.random.uniform((batch_size, diffusion_model_ddim.horizon_steps, diffusion_model_ddim.obs_dim), dtype=tf.float32)
    cond = {"state": sample_data}

    # Perform forward pass
    sample = diffusion_model_ddim(cond, deterministic=True)

    # Check output shape
    assert sample.trajectories.shape == (batch_size, diffusion_model_ddim.horizon_steps, diffusion_model_ddim.action_dim)
    assert sample.chains is None

def test_action_clipping():
    """Test the action clipping functionality of the DiffusionModel."""
    horizon_steps = 10
    obs_dim = 5
    action_dim = 3
    denoising_steps = 50
    device = "/CPU:0"  # Use CPU for testing
    final_clip_value = 0.5

    # Initialize model with action clipping
    mock_network_instance = MockNetwork()
    model_clipped = DiffusionModel(
        network=mock_network_instance,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        final_action_clip_value=final_clip_value,
        predict_epsilon=True,
        use_ddim=False
    )

    # Create mock conditional input
    batch_size = 4
    sample_data = tf.random.uniform((batch_size, horizon_steps, obs_dim), dtype=tf.float32)
    cond = {"state": sample_data}

    # Perform forward pass
    sample = model_clipped(cond, deterministic=True)

    # Check if final actions are clipped, allowing a small tolerance
    tolerance = 1e-5
    assert tf.reduce_max(sample.trajectories).numpy() <= final_clip_value + tolerance
    assert tf.reduce_min(sample.trajectories).numpy() >= -final_clip_value - tolerance

def test_denoised_clipping():
    """Test the denoised clipping functionality of the DiffusionModel."""
    horizon_steps = 10
    obs_dim = 5
    action_dim = 3
    denoising_steps = 50
    device = "/CPU:0"  # Use CPU for testing
    denoised_clip = 1.0

    # Initialize model with denoised_clip_value
    mock_network_instance = MockNetwork()
    model_denoised_clipped = DiffusionModel(
        network=mock_network_instance,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        denoised_clip_value=denoised_clip,
        predict_epsilon=True,
        use_ddim=False
    )

    # Create mock conditional input
    batch_size = 4
    sample_data = tf.random.uniform((batch_size, horizon_steps, obs_dim), dtype=tf.float32)
    cond = {"state": sample_data}

    # Perform forward pass
    sample = model_denoised_clipped(cond, deterministic=True)

    # Check if denoised_clip_value is effective, allowing a small tolerance
    tolerance = 1e-4
    max_val = tf.reduce_max(sample.trajectories).numpy()
    min_val = tf.reduce_min(sample.trajectories).numpy()
    assert max_val <= denoised_clip + tolerance, f"Max value {max_val} exceeds {denoised_clip} + tolerance"
    assert min_val >= -denoised_clip - tolerance, f"Min value {min_val} is below {-denoised_clip} - tolerance"