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
    Fixture creating a TensorFlow stitched sequence dataset.
    """
    return TFDataset(**dataset_params)

def test_dataset_lengths(torch_dataset, tf_dataset):
    """
    Test if both datasets have the same length.
    """
    assert len(torch_dataset) == len(tf_dataset), \
        f"Dataset lengths don't match: PyTorch={len(torch_dataset)}, TF={len(tf_dataset)}"

def test_indices_match(torch_dataset, tf_dataset):
    """
    Test if both datasets generate the same indices.
    """
    assert torch_dataset.indices == tf_dataset.indices, \
        "Generated indices don't match between PyTorch and TensorFlow versions"

def test_batch_shapes(torch_dataset, tf_dataset):
    """
    Test if batch shapes match between PyTorch and TensorFlow versions.
    """
    torch_batch = torch_dataset[0]
    tf_batch = tf_dataset[0]
    
    # Check actions shape
    assert torch_batch.actions.shape == tuple(tf_batch.actions.shape), \
        f"Action shapes don't match: PyTorch={torch_batch.actions.shape}, TF={tf_batch.actions.shape}"
    
    # Check conditions state shape
    assert torch_batch.conditions['state'].shape == tuple(tf_batch.conditions['state'].shape), \
        f"State shapes don't match: PyTorch={torch_batch.conditions['state'].shape}, TF={tf_batch.conditions['state'].shape}"

def test_batch_values(torch_dataset, tf_dataset):
    """
    Test if batch values are approximately equal between PyTorch and TensorFlow versions.
    """
    torch_batch = torch_dataset[0]
    tf_batch = tf_dataset[0]
    
    # Convert PyTorch tensors to numpy for comparison
    torch_actions = torch_batch.actions.numpy()
    torch_states = torch_batch.conditions['state'].numpy()
    
    # Convert TF tensors to numpy for comparison
    tf_actions = tf_batch.actions.numpy()
    tf_states = tf_batch.conditions['state'].numpy()
    
    # Compare values with small tolerance
    np.testing.assert_allclose(torch_actions, tf_actions, rtol=1e-5, atol=1e-5,
                              err_msg="Action values don't match between PyTorch and TensorFlow")
    np.testing.assert_allclose(torch_states, tf_states, rtol=1e-5, atol=1e-5,
                              err_msg="State values don't match between PyTorch and TensorFlow")

def test_multiple_batches(torch_dataset, tf_dataset):
    """
    Test multiple random batches for consistency.
    """
    for idx in [0, len(torch_dataset)//2, len(torch_dataset)-1]:  # Test start, middle, and end
        torch_batch = torch_dataset[idx]
        tf_batch = tf_dataset[idx]
        
        torch_actions = torch_batch.actions.numpy()
        torch_states = torch_batch.conditions['state'].numpy()
        
        tf_actions = tf_batch.actions.numpy()
        tf_states = tf_batch.conditions['state'].numpy()
        
        np.testing.assert_allclose(torch_actions, tf_actions, rtol=1e-5, atol=1e-5,
                                 err_msg=f"Action values don't match at index {idx}")
        np.testing.assert_allclose(torch_states, tf_states, rtol=1e-5, atol=1e-5,
                                 err_msg=f"State values don't match at index {idx}")

def test_state_history(torch_dataset, tf_dataset):
    """
    Test if conditional state history is handled correctly.
    """
    torch_batch = torch_dataset[0]
    tf_batch = tf_dataset[0]
    
    assert torch_batch.conditions['state'].shape[0] == tf_batch.conditions['state'].shape[0] == 1, \
        "Conditional state history length doesn't match specified cond_steps"

def test_action_horizon(torch_dataset, tf_dataset):
    """
    Test if action horizon is handled correctly.
    """
    torch_batch = torch_dataset[0]
    tf_batch = tf_dataset[0]
    
    assert torch_batch.actions.shape[0] == tf_batch.actions.shape[0] == 4, \
        "Action horizon doesn't match specified horizon_steps"

if __name__ == "__main__":
    pytest.main([__file__])