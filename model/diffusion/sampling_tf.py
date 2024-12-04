import tensorflow as tf
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008, dtype=tf.float32):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Converted from PyTorch to TensorFlow
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return tf.convert_to_tensor(betas_clipped, dtype=dtype)

def extract(a, t, x_shape):
    """
    Equivalent to PyTorch's gather operation for this specific use case
    Args:
        a: tensor of shape [T]
        t: tensor of shape [B]
        x_shape: tuple representing the target shape
    Returns:
        tensor of shape [B, 1, 1, ...]
    """
    b = tf.shape(t)[0]
    
    # Reshape 'a' to [1, T] and 't' to [B, 1]
    a = tf.reshape(a, [1, -1])
    t = tf.cast(t, tf.int32)
    t = tf.reshape(t, [b, 1])
    
    # Use tf.gather with batch_dims=0
    out = tf.gather(a, t, batch_dims=0, axis=1)
    
    # Reshape to match PyTorch's behavior
    return tf.reshape(out, [b] + [1] * (len(x_shape) - 1))

def make_timesteps(batch_size, i, device='CPU:0'):
    """
    Create timesteps tensor
    Note: device parameter kept for API compatibility but uses TF device placement
    """
    with tf.device(device):
        t = tf.fill([batch_size], i)
        return tf.cast(t, tf.int64)