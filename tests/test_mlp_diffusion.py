import pytest
import torch
import tensorflow as tf
import numpy as np
from torch import nn
import math

# Assuming your TF implementation is named TFDiffusionMLP
from model.diffusion.mlp_diffusion_tf import DiffusionMLP as TFDiffusionMLP
from model.diffusion.mlp_diffusion import DiffusionMLP as TorchDiffusionMLP


def test_diffusion_mlp_shapes():
    """Test if both implementations maintain expected output shapes"""

    # Test parameters
    batch_size = 8
    action_dim = 4
    horizon_steps = 10
    cond_dim = 16
    time_dim = 8
    mlp_dims = [128, 128]

    # Initialize both models
    torch_model = TorchDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=mlp_dims,
    )

    tf_model = TFDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=mlp_dims,
    )

    # Create test inputs
    x_np = np.random.normal(0, 1, (batch_size, horizon_steps, action_dim))
    time_np = np.random.randint(0, 1000, (batch_size,))
    cond_state = np.random.normal(0, 1, (batch_size, 1, cond_dim))

    # Convert to respective formats
    x_torch = torch.FloatTensor(x_np)
    time_torch = torch.LongTensor(time_np)
    cond_torch = {"state": torch.FloatTensor(cond_state)}

    x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
    time_tf = tf.convert_to_tensor(time_np, dtype=tf.int64)
    cond_tf = {"state": tf.convert_to_tensor(cond_state, dtype=tf.float32)}

    # Get outputs
    with torch.no_grad():
        torch_output = torch_model(x_torch, time_torch, cond_torch)
    tf_output = tf_model(x_tf, time_tf, cond_tf)

    # Test shapes
    assert torch_output.shape == tf_output.shape
    assert torch_output.shape == (batch_size, horizon_steps, action_dim)


def test_diffusion_mlp_extreme_times():
    """Test model behavior with extreme time values"""

    batch_size = 4
    action_dim = 2
    horizon_steps = 5
    cond_dim = 8
    time_dim = 8

    torch_model = TorchDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=[128, 128],
    )

    tf_model = TFDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=[128, 128],
    )

    # Test inputs
    x_np = np.random.normal(0, 1, (batch_size, horizon_steps, action_dim))
    cond_state = np.random.normal(0, 1, (batch_size, 1, cond_dim))

    # Test with extreme time values
    extreme_times = [0, 999]
    outputs = {}

    for t in extreme_times:
        time_np = np.full((batch_size,), t)

        x_torch = torch.FloatTensor(x_np)
        time_torch = torch.LongTensor(time_np)
        cond_torch = {"state": torch.FloatTensor(cond_state)}

        x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
        time_tf = tf.convert_to_tensor(time_np, dtype=tf.int64)
        cond_tf = {"state": tf.convert_to_tensor(cond_state, dtype=tf.float32)}

        with torch.no_grad():
            torch_output = torch_model(x_torch, time_torch, cond_torch).numpy()
        tf_output = tf_model(x_tf, time_tf, cond_tf).numpy()

        outputs[t] = {"torch": torch_output, "tf": tf_output}

    # Calculate difference between extreme times
    torch_extreme_diff = np.mean(np.abs(outputs[0]["torch"] - outputs[999]["torch"]))
    tf_extreme_diff = np.mean(np.abs(outputs[0]["tf"] - outputs[999]["tf"]))

    min_expected_diff = 0.001
    assert (
        torch_extreme_diff > min_expected_diff
    ), f"Torch extreme difference ({torch_extreme_diff:.6f}) is too small"
    assert (
        tf_extreme_diff > min_expected_diff
    ), f"TF extreme difference ({tf_extreme_diff:.6f}) is too small"


def test_diffusion_mlp_equivalence():
    """Test the equivalence between Torch and TensorFlow implementations of DiffusionMLP."""

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)

    # Model parameters
    action_dim = 2
    horizon_steps = 3
    cond_dim = 4
    time_dim = 16
    mlp_dims = [256, 256]
    batch_size = 2

    # Initialize both models
    torch_model = TorchDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=mlp_dims,
        test_mode=True,
    )

    tf_model = TFDiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        mlp_dims=mlp_dims,
        test_mode=True,
    )

    # Create test inputs
    x_np = np.random.randn(batch_size, horizon_steps, action_dim).astype(np.float32)
    time_np = np.random.randint(0, 1000, (batch_size,)).astype(np.int32)
    state_np = np.random.randn(batch_size, 1, cond_dim).astype(np.float32)

    # Convert to PyTorch tensors
    x_torch = torch.from_numpy(x_np)
    time_torch = torch.from_numpy(time_np)
    cond_torch = {"state": torch.from_numpy(state_np)}

    # Convert to TensorFlow tensors
    x_tf = tf.convert_to_tensor(x_np)
    time_tf = tf.convert_to_tensor(time_np)
    cond_tf = {"state": tf.convert_to_tensor(state_np)}

    # Get outputs
    torch_model.eval()  # Set to evaluation mode
    with torch.no_grad():
        torch_output = torch_model(x_torch, time_torch, cond_torch)

    tf_output = tf_model(x_tf, time_tf, cond_tf, training=False)

    # Convert outputs to numpy for comparison
    torch_output_np = torch_output.numpy()
    tf_output_np = tf_output.numpy()

    # Test for similar statistical properties
    np.testing.assert_allclose(torch_output_np, tf_output_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
