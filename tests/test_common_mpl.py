"""
Unit tests for the Multi-Layer Perceptron (MLP) models implemented in PyTorch and TensorFlow.

This module tests the functionality, weight transfer, and consistency between PyTorch and TensorFlow MLP implementations.
"""

import pytest
import numpy as np
import torch
import tensorflow as tf
from model.common.mlp import MLP as TorchMLP, ResidualMLP as TorchResidualMLP
from model.common.mlp_tf import MLP as TFMLP, ResidualMLP as TFResidualMLP

@pytest.fixture
def random_seed():
    """
    Fixture to set random seeds for numpy, PyTorch, and TensorFlow to ensure reproducible tests.

    Sets the seed to 42 for all libraries.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

def transfer_weights_torch_to_tf(torch_mlp, tf_mlp, input_shape):
    """
    Transfer weights from a PyTorch MLP model to a TensorFlow MLP model.

    Args:
        torch_mlp (TorchMLP): The PyTorch MLP model whose weights are to be transferred.
        tf_mlp (TFMLP): The TensorFlow MLP model to receive the weights.
        input_shape (tuple): The shape of the input data used to build the TensorFlow model.
    """
    # Build the TF model first by passing a dummy input
    dummy_input = tf.random.normal(input_shape)
    _ = tf_mlp(dummy_input)  # This builds the model
    
    tf_dense_idx = 0
    for torch_module in torch_mlp.moduleList:
        for layer in torch_module:
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.data.numpy().T
                bias = layer.bias.data.numpy()
                while tf_dense_idx < len(tf_mlp.module_list) and not isinstance(tf_mlp.module_list[tf_dense_idx], tf.keras.layers.Dense):
                    tf_dense_idx += 1
                if tf_dense_idx < len(tf_mlp.module_list):
                    tf_mlp.module_list[tf_dense_idx].set_weights([weight, bias])
                    tf_dense_idx += 1

def test_mlp_basic_forward(random_seed):
    """
    Test the basic forward pass of MLP models to ensure consistency between PyTorch and TensorFlow implementations.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    dim_list = [input_dim, hidden_dim, output_dim]
    
    # Create random input
    np_input = np.random.randn(batch_size, input_dim).astype(np.float32)
    torch_input = torch.tensor(np_input)
    tf_input = tf.convert_to_tensor(np_input)
    
    # Initialize models
    torch_mlp = TorchMLP(dim_list)
    tf_mlp = TFMLP(dim_list)
    
    # Transfer weights
    transfer_weights_torch_to_tf(torch_mlp, tf_mlp, input_shape=(batch_size, input_dim))
    
    # Forward pass
    torch_output = torch_mlp(torch_input).detach().numpy()
    tf_output = tf_mlp(tf_input).numpy()
    
    # Check outputs are similar
    np.testing.assert_allclose(torch_output, tf_output, rtol=1e-5, atol=1e-5)

def test_mlp_with_layernorm(random_seed):
    """
    Test MLP models with Layer Normalization to ensure output shapes match between PyTorch and TensorFlow.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    dim_list = [10, 20, 5]
    
    torch_mlp = TorchMLP(dim_list, use_layernorm=True)
    tf_mlp = TFMLP(dim_list, use_layernorm=True)
    
    # Test with random input
    x = np.random.randn(batch_size, dim_list[0]).astype(np.float32)
    
    # Build TF model
    _ = tf_mlp(tf.convert_to_tensor(x))
    
    torch_output = torch_mlp(torch.tensor(x)).detach().numpy()
    tf_output = tf_mlp(tf.convert_to_tensor(x)).numpy()
    
    assert torch_output.shape == tf_output.shape

def test_residual_mlp(random_seed):
    """
    Test Residual MLP models to ensure output shapes match between PyTorch and TensorFlow implementations.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    dim_list = [10, 20, 20, 20, 5]  # Input, hidden, hidden, hidden, output
    
    torch_model = TorchResidualMLP(dim_list)
    tf_model = TFResidualMLP(dim_list)
    
    x = np.random.randn(batch_size, dim_list[0]).astype(np.float32)
    
    # Build TF model
    _ = tf_model(tf.convert_to_tensor(x))
    
    torch_output = torch_model(torch.tensor(x)).detach().numpy()
    tf_output = tf_model(tf.convert_to_tensor(x)).numpy()
    
    assert torch_output.shape == tf_output.shape

def test_mlp_with_append(random_seed):
    """
    Test MLP models with appended data to ensure output shapes match between PyTorch and TensorFlow.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    dim_list = [10, 20, 5]
    append_dim = 3
    append_layers = [0]
    
    torch_mlp = TorchMLP(dim_list, append_dim=append_dim, append_layers=append_layers)
    tf_mlp = TFMLP(dim_list, append_dim=append_dim, append_layers=append_layers)
    
    x = np.random.randn(batch_size, dim_list[0]).astype(np.float32)
    append_data = np.random.randn(batch_size, append_dim).astype(np.float32)
    
    # Build TF model
    _ = tf_mlp(tf.convert_to_tensor(x), tf.convert_to_tensor(append_data))
    
    torch_output = torch_mlp(torch.tensor(x), torch.tensor(append_data)).detach().numpy()
    tf_output = tf_mlp(tf.convert_to_tensor(x), tf.convert_to_tensor(append_data)).numpy()
    
    assert torch_output.shape == tf_output.shape

def test_different_activations(random_seed):
    """
    Test MLP models with various activation functions to ensure output shapes match between PyTorch and TensorFlow.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    activation_types = ["ReLU", "ELU", "GELU", "Tanh", "Mish", "Identity", "Softplus"]
    dim_list = [10, 20, 5]
    
    for act_type in activation_types:
        torch_mlp = TorchMLP(dim_list, activation_type=act_type)
        tf_mlp = TFMLP(dim_list, activation_type=act_type)
        
        x = np.random.randn(batch_size, dim_list[0]).astype(np.float32)
        
        # Build TF model
        _ = tf_mlp(tf.convert_to_tensor(x))
        
        torch_output = torch_mlp(torch.tensor(x)).detach().numpy()
        tf_output = tf_mlp(tf.convert_to_tensor(x)).numpy()
        
        assert torch_output.shape == tf_output.shape

def test_mlp_with_dropout(random_seed):
    """
    Test MLP models with dropout to ensure output shapes match between PyTorch and TensorFlow.

    Args:
        random_seed: Fixture that sets the random seed for reproducibility.
    """
    batch_size = 32
    dim_list = [10, 20, 5]
    dropout = 0.2
    
    torch_mlp = TorchMLP(dim_list, dropout=dropout)
    tf_mlp = TFMLP(dim_list, dropout=dropout)
    
    # Set to eval mode for torch (tensorflow doesn't need this)
    torch_mlp.eval()
    
    x = np.random.randn(batch_size, dim_list[0]).astype(np.float32)
    
    # Build TF model
    _ = tf_mlp(tf.convert_to_tensor(x))
    
    torch_output = torch_mlp(torch.tensor(x)).detach().numpy()
    tf_output = tf_mlp(tf.convert_to_tensor(x), training=False).numpy()
    
    assert torch_output.shape == tf_output.shape
    
if __name__ == "__main__":
    pytest.main([__file__])