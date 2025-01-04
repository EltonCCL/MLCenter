import pytest
import torch
import tensorflow as tf
import numpy as np
import random

# Import your EMA implementations
from agent.pretrain.train_agent import EMA as TorchEMA
from agent.pretrain.train_agent_tf import EMA as TfEMA

# Dummy configuration for testing
class DummyConfig:
    def __init__(self, decay):
        self.decay = decay

def clone_model_and_weights(model):
    """Deep clones a TensorFlow Keras model, including weights."""
    new_model = tf.keras.models.clone_model(model)
    new_model.build(model.input_shape)
    new_model.set_weights(model.get_weights())
    return new_model

@pytest.mark.parametrize("decay", [0.9, 0.99, 0.999])
def test_ema_consistency_fixed(decay):
    # 1. Set Seed for Reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    tf.random.set_seed(seed_value)

    # 2. Create Dummy Models with Fixed Weights
    torch_model = torch.nn.Linear(10, 5)
    tf_model = tf.keras.models.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])

    # Initialize TensorFlow model with the same weights as PyTorch model
    with torch.no_grad():
        torch_model.weight.data = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
                                                [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1],
                                                [-0.11, -0.22, -0.33, -0.44, -0.55, -0.66, -0.77, -0.88, -0.99, -1.1],
                                                [0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 0.90, 1.2]], dtype=torch.float32)
        torch_model.bias.data.fill_(0.0)

    tf_weights = [torch_model.weight.data.numpy().T, torch_model.bias.data.numpy()]
    tf_model.set_weights(tf_weights)

    # Create EMA updaters
    cfg = DummyConfig(decay)
    torch_ema_updater = TorchEMA(cfg)
    tf_ema_updater = TfEMA(cfg)

    # Create EMA models
    torch_ema_model = torch.nn.Linear(10, 5)
    torch_ema_model.load_state_dict(torch_model.state_dict())

    tf_ema_model = clone_model_and_weights(tf_model)

    # Build the TensorFlow models
    dummy_input = tf.zeros((1, 10))
    _ = tf_model(dummy_input)
    _ = tf_ema_model(dummy_input)

    # 3. Apply a Fixed Update (or No Update)
    # Example of a fixed update (PyTorch)
    with torch.no_grad():
        for param in torch_model.parameters():
            param.data.add_(0.01)  # Fixed update

    # Example of a fixed update (TensorFlow)
    for i in range(len(tf_model.trainable_variables)):
        tf_model.trainable_variables[i].assign_add(tf.ones_like(tf_model.trainable_variables[i]) * 0.01) # Fixed update

    # 4. Perform EMA Update
    torch_ema_updater.update_model_average(torch_ema_model, torch_model)
    tf_ema_updater.update_model_average(tf_ema_model, tf_model)

    # 5. Compare EMA Weights
    print(f"\nEMA Weights After Update (Decay = {decay}):")
    for i, (torch_ema_param, tf_ema_param) in enumerate(zip(torch_ema_model.parameters(), tf_ema_model.trainable_variables)):
        if len(torch_ema_param.shape) > 1:
            print(f"  Weight {i} (Torch):\n{torch_ema_param.data.numpy().T}")
            print(f"  Weight {i} (TF):\n{tf_ema_param.numpy()}")
            assert np.allclose(torch_ema_param.data.numpy().T, tf_ema_param.numpy(), atol=1e-7, rtol=1e-7)
        else:
            print(f"  Bias {i} (Torch):\n{torch_ema_param.data.numpy()}")
            print(f"  Bias {i} (TF):\n{tf_ema_param.numpy()}")
            assert np.allclose(torch_ema_param.data.numpy(), tf_ema_param.numpy(), atol=1e-7, rtol=1e-7)

if __name__ == "__main__":
    pytest.main([__file__])