import pytest
import numpy as np
import torch
import tensorflow as tf
from agent.dataset.sequence import StitchedSequenceDataset as TorchDataset
from agent.dataset.sequence_tf import StitchedSequenceDataset as TFDataset

@pytest.fixture
def dataset_params():
    """
    Fixture providing parameters for dataset initialization.
    """
    return {
        'dataset_path': 'data/gym/hopper-medium-v2/train.npz',
        'horizon_steps': 4,
        'cond_steps': 1,
        'max_n_episodes': 500  # Use small number for testing
    }

@pytest.fixture
def torch_dataset(dataset_params):
    """
    Fixture creating a PyTorch stitched sequence dataset.
    """
    return TorchDataset(**dataset_params, device='cpu')

@pytest.fixture
def tf_dataset(dataset_params):
    """
    Fixture creating a TensorFlow stitched sequence dataset as tf.data.Dataset.
    """
    dataset = TFDataset(**dataset_params)
    return dataset.as_tensorflow_dataset(batch_size=1, shuffle=False)

def test_dataset_lengths(torch_dataset, tf_dataset):
    """
    Test if both datasets have the same length.
    """
    assert len(torch_dataset) == len(tf_dataset), \
        f"Dataset lengths don't match: PyTorch={len(torch_dataset)}, TF={len(tf_dataset)}"

def test_batch_shapes(torch_dataset, tf_dataset):
    """
    Test if batch shapes match between PyTorch and TensorFlow versions.
    """
    torch_batch = torch_dataset[0]
    tf_batch = next(iter(tf_dataset))
    
    # Remove batch dimension from TensorFlow batch
    tf_actions = tf_batch[0][0]
    tf_state = tf_batch[1]['state'][0]
    
    # Check actions shape
    assert torch_batch.actions.shape == tuple(tf_actions.shape), \
        f"Action shapes don't match: PyTorch={torch_batch.actions.shape}, TF={tf_actions.shape}"
        
    # Check conditions state shape
    assert torch_batch.conditions['state'].shape == tuple(tf_state.shape), \
        f"State shapes don't match: PyTorch={torch_batch.conditions['state'].shape}, TF={tf_state.shape}"

def test_batch_values(torch_dataset, tf_dataset):
    """
    Test if batch values are approximately equal between PyTorch and TensorFlow versions.
    """
    torch_batch = torch_dataset[0]
    tf_batch = next(iter(tf_dataset))

    # Remove batch dimension and convert TF tensors to numpy
    tf_actions = tf_batch[0][0].numpy()
    tf_states = tf_batch[1]['state'][0].numpy()
    
    # Convert PyTorch tensors to numpy
    torch_actions = torch_batch.actions.numpy()
    torch_states = torch_batch.conditions['state'].numpy()
    
    # Compare values with small tolerance
    np.testing.assert_allclose(torch_actions, tf_actions, rtol=1e-5, atol=1e-5,
                              err_msg="Action values don't match between PyTorch and TensorFlow")
    np.testing.assert_allclose(torch_states, tf_states, rtol=1e-5, atol=1e-5,
                              err_msg="State values don't match between PyTorch and TensorFlow")

def test_multiple_batches(torch_dataset, tf_dataset):
    """
    Test multiple random batches for consistency.
    """
    indices = [0, len(torch_dataset)//2, len(torch_dataset)-1]
    for idx in indices:
        torch_batch = torch_dataset[idx]
        tf_batch = next(iter(tf_dataset.skip(idx).take(1)))

        # Remove batch dimension and convert TF tensors to numpy
        tf_actions = tf_batch[0][0].numpy()
        tf_states = tf_batch[1]['state'][0].numpy()
        
        # Convert PyTorch tensors to numpy
        torch_actions = torch_batch.actions.numpy()
        torch_states = torch_batch.conditions['state'].numpy()
        
        # Compare values with small tolerance
        np.testing.assert_allclose(torch_actions, tf_actions, rtol=1e-5, atol=1e-5,
                                 err_msg=f"Action values don't match at index {idx}")
        np.testing.assert_allclose(torch_states, tf_states, rtol=1e-5, atol=1e-5,
                                 err_msg=f"State values don't match at index {idx}")

def test_state_history(torch_dataset, tf_dataset):
    """
    Test if conditional state history is handled correctly.
    """
    torch_batch = torch_dataset[0]
    tf_batch = next(iter(tf_dataset))
    
    assert torch_batch.conditions['state'].shape[0] == tf_batch[1]['state'].shape[0] == 1, \
        "Conditional state history length doesn't match specified cond_steps"

def test_action_horizon(torch_dataset, tf_dataset):
    """
    Test if action horizon is handled correctly.
    """
    torch_batch = torch_dataset[0]
    tf_batch = next(iter(tf_dataset))
    
    # Remove batch dimension from TensorFlow batch
    tf_actions = tf_batch[0][0]  # Shape after removing batch dimension
    assert torch_batch.actions.shape[0] == tf_actions.shape[0] == 4, \
        "Action horizon doesn't match specified horizon_steps"

if __name__ == "__main__":
    pytest.main([__file__])