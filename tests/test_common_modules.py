"""Unit tests for common modules in the diffusion model."""

import pytest
import numpy as np
import torch
import tensorflow as tf
from model.common.modules import SpatialEmb as TorchSpatialEmb, RandomShiftsAug as TorchRandomShiftsAug
from model.common.modules_tf import SpatialEmb as TFSpatialEmb, RandomShiftsAug as TFRandomShiftsAug

@pytest.fixture
def random_seed():
    """Fixture to set random seed for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

@pytest.fixture
def spatial_emb_params():
    """Fixture to provide parameters for SpatialEmb."""
    return {
        'num_patch': 48,
        'patch_dim': 48,
        'prop_dim': 11,
        'proj_dim': 11,
        'dropout': 0.0
    }

class TestSpatialEmb:
    """Tests for the SpatialEmb module."""

    def copy_weights(self, tf_model, torch_model, spatial_emb_params):
        """Copies weights from PyTorch model to TensorFlow model."""
        # Build TF model by calling it once
        batch_size = 1
        dummy_feat = tf.zeros((batch_size, spatial_emb_params['patch_dim'], spatial_emb_params['num_patch']))
        dummy_prop = tf.zeros((batch_size, spatial_emb_params['prop_dim']))
        _ = tf_model(dummy_feat, dummy_prop, training=False)

        # Copy the weight parameter
        tf_model.weight.assign(torch_model.weight.detach().numpy())
        
        # Copy the weights from input_proj layers and ensure same normalization parameters
        for tf_layer, torch_layer in zip(tf_model.input_proj.layers, torch_model.input_proj):
            if isinstance(tf_layer, tf.keras.layers.Dense):
                tf_layer.set_weights([
                    torch_layer.weight.detach().numpy().T,
                    torch_layer.bias.detach().numpy()
                ])
            elif isinstance(tf_layer, tf.keras.layers.LayerNormalization):
                # Set the same epsilon value
                tf_layer.epsilon = 1e-5  # PyTorch default
                tf_layer.set_weights([
                    torch_layer.weight.detach().numpy(),
                    torch_layer.bias.detach().numpy()
                ])

    def test_output_values(self, random_seed, spatial_emb_params):
        """Tests the output values of SpatialEmb for consistency between PyTorch and TensorFlow."""
        batch_size = 8
        feat = np.random.randn(batch_size, spatial_emb_params['patch_dim'], spatial_emb_params['num_patch'])
        prop = np.random.randn(batch_size, spatial_emb_params['prop_dim'])

        # PyTorch version
        torch_model = TorchSpatialEmb(**spatial_emb_params)
        torch_model.eval()  # Set to eval mode
        torch_feat = torch.FloatTensor(feat)
        torch_prop = torch.FloatTensor(prop)
        
        # TensorFlow version
        tf_model = TFSpatialEmb(**spatial_emb_params)
        self.copy_weights(tf_model, torch_model, spatial_emb_params)

        tf_feat = tf.convert_to_tensor(feat, dtype=tf.float32)
        tf_prop = tf.convert_to_tensor(prop, dtype=tf.float32)

        # Debug intermediate values - PyTorch
        with torch.no_grad():
            # Step 1: Get input_proj intermediate outputs
            torch_feat_t = torch_feat.transpose(1, 2)
            if spatial_emb_params['prop_dim'] > 0:
                repeated_prop = torch_prop.unsqueeze(1).repeat(1, torch_feat_t.size(1), 1)
                torch_feat_cat = torch.cat((torch_feat_t, repeated_prop), dim=-1)
            else:
                torch_feat_cat = torch_feat_t
            
            # Get intermediate outputs from each layer
            torch_intermediate = torch_feat_cat
            torch_intermediates = []
            for layer in torch_model.input_proj:
                torch_intermediate = layer(torch_intermediate)
                torch_intermediates.append(torch_intermediate)

            torch_y = torch_intermediate
            torch_z = (torch_model.weight * torch_y).sum(1)
            torch_output = torch_model.dropout(torch_z)

        # Debug intermediate values - TensorFlow
        tf_feat_t = tf.transpose(tf_feat, perm=[0, 2, 1])
        if spatial_emb_params['prop_dim'] > 0:
            repeated_prop_tf = tf.expand_dims(tf_prop, axis=1)
            repeated_prop_tf = tf.tile(repeated_prop_tf, [1, tf.shape(tf_feat_t)[1], 1])
            tf_feat_cat = tf.concat([tf_feat_t, repeated_prop_tf], axis=-1)
        else:
            tf_feat_cat = tf_feat_t

        # Get intermediate outputs from each layer
        tf_intermediate = tf_feat_cat
        tf_intermediates = []
        for layer in tf_model.input_proj.layers:
            tf_intermediate = layer(tf_intermediate)
            tf_intermediates.append(tf_intermediate)

        tf_y = tf_intermediate
        tf_z = tf.reduce_sum(tf_model.weight * tf_y, axis=1)
        tf_output = tf_model(tf_feat, tf_prop, training=False)

        # Print detailed intermediate comparisons
        print("\nDetailed Layer-by-Layer Comparison:")
        for i, (torch_inter, tf_inter) in enumerate(zip(torch_intermediates, tf_intermediates)):
            max_diff = np.max(np.abs(torch_inter.numpy() - tf_inter.numpy()))
            print(f"Layer {i} max diff: {max_diff}")
            
            # If difference is significant, print statistics
            if max_diff > 0.01:
                print(f"Layer {i} stats:")
                print(f"Torch mean: {torch_inter.mean().item()}, std: {torch_inter.std().item()}")
                print(f"TF mean: {tf.reduce_mean(tf_inter).numpy()}, std: {tf.math.reduce_std(tf_inter).numpy()}")

        print("\nFinal Output Stats:")
        print(f"Torch output - mean: {torch_output.mean().item()}, std: {torch_output.std().item()}")
        print(f"TF output - mean: {tf.reduce_mean(tf_output).numpy()}, std: {tf.math.reduce_std(tf_output).numpy()}")
        print(f"Max absolute difference: {np.max(np.abs(torch_output.numpy() - tf_output.numpy()))}")

        # Try a more relaxed tolerance
        np.testing.assert_allclose(torch_output.numpy(), tf_output.numpy(), rtol=1e-4, atol=1e-6)

class TestRandomShiftsAug:
    """Tests for the RandomShiftsAug augmentation module."""

    def test_output_shape(self, random_seed):
        """Ensures the output shape is consistent after augmentation."""
        batch_size = 8
        channels = 3
        height = width = 64
        pad = 4

        # Create random image batch
        images = np.random.randint(0, 255, (batch_size, channels, height, width))

        # PyTorch version
        torch_aug = TorchRandomShiftsAug(pad=pad)
        torch_images = torch.FloatTensor(images)
        torch_output = torch_aug(torch_images)

        # TensorFlow version
        tf_aug = TFRandomShiftsAug(pad=pad)
        tf_images = tf.convert_to_tensor(images, dtype=tf.float32)
        tf_output = tf_aug(tf_images)

        # Check output shapes
        assert torch_output.shape == (batch_size, channels, height, width)
        assert torch_output.shape == tf_output.shape

def test_gradient_flow(random_seed, spatial_emb_params):
    """Tests the gradient flow for both PyTorch and TensorFlow implementations."""
    batch_size = 8
    feat = np.random.randn(batch_size, spatial_emb_params['patch_dim'], spatial_emb_params['num_patch'])
    prop = np.random.randn(batch_size, spatial_emb_params['prop_dim'])

    # PyTorch gradient test
    torch_model = TorchSpatialEmb(**spatial_emb_params)
    torch_feat = torch.FloatTensor(feat).requires_grad_()
    torch_prop = torch.FloatTensor(prop)
    
    torch_output = torch_model(torch_feat, torch_prop)
    torch_loss = torch.mean(torch_output)  # Simple loss for gradient testing
    torch_loss.backward()
    
    assert torch_feat.grad is not None
    assert not torch.isnan(torch_feat.grad).any()

    # TensorFlow gradient test
    tf_model = TFSpatialEmb(**spatial_emb_params)
    with tf.GradientTape() as tape:
        tf_feat = tf.convert_to_tensor(feat, dtype=tf.float32)
        tape.watch(tf_feat)
        tf_prop = tf.convert_to_tensor(prop, dtype=tf.float32)
        
        tf_output = tf_model(tf_feat, tf_prop)
        tf_loss = tf.reduce_mean(tf_output)
    
    tf_grads = tape.gradient(tf_loss, tf_feat)
    
    assert tf_grads is not None
    assert not tf.math.reduce_any(tf.math.is_nan(tf_grads))

if __name__ == "__main__":
    pytest.main([__file__])