# project/tests/test_layers.py

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from model.diffusion.modules import (
    SinusoidalPosEmb as SinusoidalPosEmbTorch,
    Downsample1d as Downsample1dTorch,
    Upsample1d as Upsample1dTorch,
    Conv1dBlock as Conv1dBlockTorch
)
from model.diffusion.modules_tf import (
    SinusoidalPosEmb as SinusoidalPosEmbTF,
    Downsample1d as Downsample1dTF,
    Upsample1d as Upsample1dTF,
    Conv1dBlock as Conv1dBlockTF
)

@pytest.fixture(scope='session')
def torch_device():
    return torch.device('cpu')

@pytest.fixture(scope='session', autouse=True)
def random_seed():
    torch.manual_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

@pytest.fixture
def sinusoidal_input():
    def _create(batch_size=2):
        torch_input = torch.linspace(0, 1, steps=batch_size).to(torch.float32)
        tf_input = tf.linspace(0.0, 1.0, batch_size)
        return torch_input, tf_input
    return _create

@pytest.fixture
def conv1d_input():
    def _create(batch_size=2, channels=8, length=16):
        torch_input = torch.ones((batch_size, channels, length), dtype=torch.float32)
        tf_input = tf.ones((batch_size, length, channels), dtype=tf.float32)
        return torch_input, tf_input
    return _create

class TestSinusoidalPosEmb:
    """Tests for SinusoidalPosEmb layers."""

    def test_output_matches(self, torch_device, sinusoidal_input):
        """
        Test that the Torch and TensorFlow SinusoidalPosEmb outputs match.

        Args:
            torch_device (torch.device): The device to run Torch operations on.
            sinusoidal_input (callable): Function to create sinusoidal inputs.
        """
        # Arrange
        dim = 16
        batch_size = 4
        torch_layer = SinusoidalPosEmbTorch(dim).to(torch_device).eval()
        tf_layer = SinusoidalPosEmbTF(dim)
        tf_layer.build(input_shape=(None,))
        x_torch, x_tf = sinusoidal_input(batch_size)
        x_torch = x_torch.to(torch_device)

        # Act
        with torch.no_grad():
            emb_torch = torch_layer(x_torch).cpu().numpy()
        emb_tf = tf_layer(x_tf).numpy()

        # Assert
        np.testing.assert_allclose(emb_torch, emb_tf, rtol=1e-5, atol=1e-5)

class TestDownsample1d:
    """Tests for Downsample1d layers."""

    @pytest.fixture
    def model_inputs(self):
        """
        Fixture to create model inputs for Downsample1d tests.

        Returns:
            callable: Function to create inputs with specified batch size, sequence length, and channels.
        """
        def _create(batch_size=2, sequence_length=10, channels=4):
            x_numpy = np.random.randn(batch_size, sequence_length, channels)
            x_torch = torch.FloatTensor(x_numpy.transpose(0, 2, 1))
            x_tf = tf.convert_to_tensor(x_numpy, dtype=tf.float32)
            return x_numpy, x_torch, x_tf
        return _create

    def test_output_matches(self, model_inputs, random_seed):
        """
        Test that the Torch and TensorFlow Downsample1d outputs match.

        Args:
            model_inputs (callable): Function to create model inputs.
            random_seed (None): Fixture to set random seed.
        """
        # Arrange
        channels = 4
        x_numpy, x_torch, x_tf = model_inputs()
        
        torch_model = Downsample1dTorch(channels)
        tf_model = Downsample1dTF(channels)
        _ = tf_model(tf.random.normal((1, x_numpy.shape[1], channels)))
        
        # Set TF weights to match PyTorch
        torch_weight = torch_model.conv.weight.data.numpy()
        torch_bias = torch_model.conv.bias.data.numpy()
        tf_weight = torch_weight.transpose(2, 1, 0)
        tf_model.conv.set_weights([tf_weight, torch_bias])

        # Act
        with torch.no_grad():
            torch_output = torch_model(x_torch).numpy().transpose(0, 2, 1)
        tf_output = tf_model(x_tf).numpy()

        # Assert
        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-4, atol=1e-4)

class TestUpsample1d:
    """Tests for Upsample1d layers."""

    @pytest.fixture
    def model_inputs(self):
        """
        Fixture to create model inputs for Upsample1d tests.

        Returns:
            callable: Function to create inputs with specified batch size, sequence length, and channels.
        """
        def _create(batch_size=2, sequence_length=10, channels=4):
            x_numpy = np.random.randn(batch_size, sequence_length, channels)
            x_torch = torch.FloatTensor(x_numpy.transpose(0, 2, 1))
            x_tf = tf.convert_to_tensor(x_numpy, dtype=tf.float32)
            return x_numpy, x_torch, x_tf
        return _create

    def test_output_matches(self, model_inputs, random_seed):
        """
        Test that the Torch and TensorFlow Upsample1d outputs match.

        Args:
            model_inputs (callable): Function to create model inputs.
            random_seed (None): Fixture to set random seed.
        """
        # Arrange
        channels = 4
        x_numpy, x_torch, x_tf = model_inputs()
        
        torch_model = Upsample1dTorch(channels)
        tf_model = Upsample1dTF(channels)
        _ = tf_model(tf.random.normal((1, x_numpy.shape[1], channels)))

        # Set TF weights to match PyTorch
        torch_weight = torch_model.conv.weight.data.numpy()
        torch_bias = torch_model.conv.bias.data.numpy()
        tf_weight = torch_weight.transpose(2, 1, 0).reshape(4, 1, channels, channels)
        tf_model.conv.set_weights([tf_weight, torch_bias])

        # Act
        with torch.no_grad():
            torch_output = torch_model(x_torch).numpy().transpose(0, 2, 1)
        tf_output = tf_model(x_tf).numpy()

        # Assert
        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-4, atol=1e-4)

class TestConv1dBlock:
    """Tests for Conv1dBlock layers."""

    @pytest.mark.parametrize(
        "batch_size,sequence_length,in_channels,out_channels,kernel_size,n_groups",
        [
            (2, 16, 4, 8, 3, None),  # No group norm
            (2, 16, 4, 8, 3, 2),     # With group norm
        ]
    )
    def test_output_matches(self, batch_size, sequence_length, in_channels, 
                            out_channels, kernel_size, n_groups, random_seed):
        """
        Test that the Torch and TensorFlow Conv1dBlock outputs match.

        Args:
            batch_size (int): Number of samples in a batch.
            sequence_length (int): Length of the input sequence.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            n_groups (int or None): Number of groups for GroupNormalization.
            random_seed (None): Fixture to set random seed.
        """
        # Arrange
        x_numpy = np.random.randn(batch_size, sequence_length, in_channels)
        x_torch = torch.FloatTensor(x_numpy.transpose(0, 2, 1))
        x_tf = tf.convert_to_tensor(x_numpy, dtype=tf.float32)

        torch_model = Conv1dBlockTorch(in_channels, out_channels, kernel_size, n_groups)
        tf_model = Conv1dBlockTF(in_channels, out_channels, kernel_size, n_groups)
        _ = tf_model(tf.random.normal((1, sequence_length, in_channels)))

        # Set weights
        torch_conv_weight = torch_model.block[0].weight.data.numpy()
        torch_conv_bias = torch_model.block[0].bias.data.numpy()
        tf_weight = torch_conv_weight.transpose(2, 1, 0)
        tf_model.conv.set_weights([tf_weight, torch_conv_bias])

        if n_groups is not None:
            torch_gn_weight = torch_model.block[2].weight.data.numpy()
            torch_gn_bias = torch_model.block[2].bias.data.numpy()
            tf_model.gn.set_weights([torch_gn_weight, torch_gn_bias])

        # Act
        with torch.no_grad():
            torch_output = torch_model(x_torch).numpy().transpose(0, 2, 1)
        tf_output = tf_model(x_tf).numpy()

        # Assert
        np.testing.assert_allclose(torch_output, tf_output, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])