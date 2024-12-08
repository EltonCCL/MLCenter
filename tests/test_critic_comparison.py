import pytest
import torch
import tensorflow as tf
import numpy as np
from model.common.critic import CriticObs as TorchCriticObs, CriticObsAct as TorchCriticObsAct
from model.common.critic_tf import CriticObs as TFCriticObs, CriticObsAct as TFCriticObsAct

@pytest.fixture
def random_seed():
    """
    Fixture to set random seeds for reproducibility in tests.
    """
    torch.manual_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)

def test_critic_obs_test_mode(random_seed):
    """
    Test CriticObs models in test mode for consistent outputs between PyTorch and TensorFlow implementations.
    """
    # Initialize models
    cond_dim = 64
    torch_critic = TorchCriticObs(cond_dim=cond_dim, mlp_dims=[128, 64], test_mode=True)
    tf_critic = TFCriticObs(cond_dim=cond_dim, mlp_dims=[128, 64], test_mode=True)
    
    # Build TF model by calling it once
    batch_size = 32
    dummy_input = tf.random.normal((batch_size, cond_dim))
    _ = tf_critic(dummy_input)
    
    # Copy weights from PyTorch to TensorFlow
    torch_weight = torch_critic.Q1.weight.data.numpy()
    torch_bias = torch_critic.Q1.bias.data.numpy()
    
    tf_critic.Q1.kernel.assign(torch_weight.T)
    tf_critic.Q1.bias.assign(torch_bias)
    
    # Create test input
    torch_input = torch.randn(batch_size, cond_dim)
    tf_input = tf.convert_to_tensor(torch_input.numpy())
    
    # Get outputs
    torch_output = torch_critic(torch_input)
    tf_output = tf_critic(tf_input)
    
    # Compare outputs
    np.testing.assert_allclose(
        torch_output.detach().numpy(), 
        tf_output.numpy(), 
        rtol=1e-5, atol=1e-5
    )

def test_critic_obs_regular_mode(random_seed):
    """
    Test CriticObs models in regular mode with various configurations to ensure output shapes match between PyTorch and TensorFlow.
    """
    configs = [
        {"cond_dim": 64, "mlp_dims": [128, 64], "use_layernorm": False},
        {"cond_dim": 32, "mlp_dims": [64, 32], "use_layernorm": True},
        {"cond_dim": 128, "mlp_dims": [256, 128, 64], "residual_style": True},
    ]
    
    for config in configs:
        torch_critic = TorchCriticObs(**config, test_mode=False)
        tf_critic = TFCriticObs(**config, test_mode=False)
        
        # Test with both dictionary and tensor inputs
        batch_size = 16
        
        # Dictionary input
        time_steps = 4
        state_dim = config["cond_dim"] // time_steps  # Adjust state_dim so that when flattened it matches cond_dim
        torch_dict_input = {
            "state": torch.randn(batch_size, time_steps, state_dim)
        }
        tf_dict_input = {
            "state": tf.convert_to_tensor(torch_dict_input["state"].numpy())
        }
        
        # Tensor input
        torch_tensor_input = torch.randn(batch_size, config["cond_dim"])
        tf_tensor_input = tf.convert_to_tensor(torch_tensor_input.numpy())
        
        # Check output shapes
        torch_dict_output = torch_critic(torch_dict_input)
        tf_dict_output = tf_critic(tf_dict_input)
        assert torch_dict_output.shape == tf_dict_output.shape
        
        torch_tensor_output = torch_critic(torch_tensor_input)
        tf_tensor_output = tf_critic(tf_tensor_input)
        assert torch_tensor_output.shape == tf_tensor_output.shape

def test_critic_obs_act_test_mode(random_seed):
    """
    Test CriticObsAct models in test mode, verifying weight transfer and output consistency between PyTorch and TensorFlow implementations.
    """
    # Initialize models
    cond_dim = 64
    action_dim = 32
    torch_critic = TorchCriticObsAct(
        cond_dim=cond_dim, 
        mlp_dims=[128, 64], 
        action_dim=action_dim,
        test_mode=True
    )
    tf_critic = TFCriticObsAct(
        cond_dim=cond_dim, 
        mlp_dims=[128, 64], 
        action_dim=action_dim,
        test_mode=True
    )
    
    # Build TF model by calling it once
    batch_size = 32
    time_steps = 2
    state_dim = cond_dim // time_steps
    dummy_state = tf.random.normal((batch_size, time_steps, state_dim))
    dummy_action = tf.random.normal((batch_size, time_steps, action_dim))
    _ = tf_critic({"state": dummy_state}, dummy_action)
    
    # Copy weights from PyTorch to TensorFlow for Q1
    torch_weight_q1 = torch_critic.Q1.weight.data.numpy()
    torch_bias_q1 = torch_critic.Q1.bias.data.numpy()
    tf_critic.Q1.kernel.assign(torch_weight_q1.T)
    tf_critic.Q1.bias.assign(torch_bias_q1)
    
    # Copy weights from PyTorch to TensorFlow for Q2 if exists
    if hasattr(torch_critic, 'Q2') and hasattr(tf_critic, 'Q2'):
        torch_weight_q2 = torch_critic.Q2.weight.data.numpy()
        torch_bias_q2 = torch_critic.Q2.bias.data.numpy()
        tf_critic.Q2.kernel.assign(torch_weight_q2.T)
        tf_critic.Q2.bias.assign(torch_bias_q2)
    
    # Create test input
    torch_input = torch.randn(batch_size, time_steps, state_dim)
    torch_action = torch.randn(batch_size, time_steps, action_dim)
    
    # Convert inputs to TensorFlow tensors
    tf_input = tf.convert_to_tensor(torch_input.numpy(), dtype=tf.float32)
    tf_action = tf.convert_to_tensor(torch_action.numpy(), dtype=tf.float32)
    
    # Get outputs
    torch_output = torch_critic({"state": torch_input}, torch_action)
    tf_output = tf_critic({"state": tf_input}, tf_action)
    
    # Compare outputs
    if isinstance(torch_output, tuple):
        for t_out, tf_out in zip(torch_output, tf_output):
            np.testing.assert_allclose(
                t_out.detach().numpy(),
                tf_out.numpy(),
                rtol=1e-5, atol=1e-5
            )
    else:
        np.testing.assert_allclose(
            torch_output.detach().numpy(),
            tf_output.numpy(),
            rtol=1e-5, atol=1e-5
        )

if __name__ == "__main__":
    """
    Entry point for running the tests using pytest.
    """
    pytest.main([__file__])