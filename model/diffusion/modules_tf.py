import math
import tensorflow as tf
from tensorflow.keras import layers

# Custom Mish Activation Layer
class Mish(layers.Layer):
    """
    Custom Mish Activation Layer.

    Args:
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, x):
        """
        Apply the Mish activation function.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Activated tensor.
        """
        return x * tf.math.tanh(tf.math.softplus(x))

    def get_config(self):
        """
        Return the config of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(Mish, self).get_config()
        return config

# Sinusoidal Positional Embedding Layer
class SinusoidalPosEmb(layers.Layer):
    """
    Sinusoidal Positional Embedding Layer.

    Args:
        dim (int): Dimension of the embedding.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, dim, **kwargs):
        super(SinusoidalPosEmb, self).__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        """
        Compute sinusoidal embeddings for input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch,) or (batch, 1).

        Returns:
            tf.Tensor: Embedding tensor of shape (batch, dim).
        """
        half_dim = self.dim // 2
        emb_scaling = math.log(10000.0) / (half_dim - 1)
        emb = tf.range(half_dim, dtype=x.dtype) * -emb_scaling
        emb = tf.exp(emb)  # Shape: (half_dim,)
        emb = tf.expand_dims(x, axis=-1) * tf.expand_dims(emb, axis=0)  # Shape: (batch, half_dim)
        emb_sin = tf.sin(emb)
        emb_cos = tf.cos(emb)
        emb = tf.concat([emb_sin, emb_cos], axis=-1)  # Shape: (batch, dim)
        return emb

    def get_config(self):
        """
        Return the config of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(SinusoidalPosEmb, self).get_config()
        config.update({
            'dim': self.dim
        })
        return config
    
# class Downsample1d(layers.Layer):
#     def __init__(self, dim, **kwargs):
#         super(Downsample1d, self).__init__(**kwargs)
#         self.conv = layers.Conv1D(
#             filters=dim,
#             kernel_size=3,
#             strides=2,
#             padding='valid',
#             use_bias=True,
#         )

#     def call(self, x):
#         """
#         x: Tensor of shape (batch, length, channels)
#         Returns:
#             Downsampled tensor of shape (batch, ceil(length/2), channels)
#         """
#         return self.conv(x)

#     def get_config(self):
#         config = super(Downsample1d, self).get_config()
#         config.update({
#             'dim': self.conv.filters
#         })
#         return config
# Downsampling Layer using Conv1D
class Downsample1d(layers.Layer):
    """
    Downsampling Layer using Conv1D.

    Args:
        dim (int): Number of output filters.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, dim, **kwargs):
        super(Downsample1d, self).__init__(**kwargs)
        self.conv = layers.Conv1D(
            filters=dim,
            kernel_size=3,
            strides=2,
            padding='valid'  # Changed from 'same' to 'valid'
        )
        self.pad = layers.ZeroPadding1D(padding=(1, 1))  # Explicit padding of 1 on each side

    def call(self, x):
        """
        Downsample the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, length, channels).

        Returns:
            tf.Tensor: Downsampled tensor of shape (batch, ceil(length/2), channels).
        """
        # x shape: (batch, length, channels)
        x = self.pad(x)  # Add padding first
        return self.conv(x)  # Then apply convolution
    
class Upsample1d(tf.keras.Model):
    """
    Upsampling Layer using Conv2DTranspose.

    Args:
        dim (int): Number of output filters.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=dim,
            kernel_size=(4, 1),
            strides=(2, 1),
            padding='same',
        )
        
    def call(self, x):
        """
        Upsample the input tensor.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Upsampled tensor.
        """
        # Add dummy dimension for height
        x = tf.expand_dims(x, axis=2)
        # Apply transposed convolution
        x = self.conv(x)
        # Remove dummy dimension
        x = tf.squeeze(x, axis=2)
        return x

# Convolutional Block with Optional GroupNormalization and Activation
class Conv1dBlock(layers.Layer):
    """
    Convolutional Block with Optional GroupNormalization and Activation.

    Args:
        inp_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        n_groups (int, optional): Number of groups for GroupNormalization.
        activation_type (str): Type of activation ('Mish' or 'ReLU').
        eps (float): Epsilon for numerical stability in normalization.
        **kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
        **kwargs
    ):
        super(Conv1dBlock, self).__init__(**kwargs)
        self.layers_list = []
        
        # Conv1D Layer
        self.conv = layers.Conv1D(
            filters=out_channels,
            kernel_size=kernel_size,
            padding='same',  # Equivalent to padding=kernel_size//2 in PyTorch for odd kernel sizes
            # use_bias=False  # Typically, normalization layers handle the bias
        )
        self.layers_list.append(self.conv)

        # GroupNormalization Layer (optional)
        if n_groups is not None:
            self.gn = layers.GroupNormalization(
                groups=n_groups,
                axis=-1,  # Channels last
                epsilon=eps
            )
            self.layers_list.append(self.gn)

        # Activation Layer
        if activation_type == "Mish":
            self.activation = Mish()
        elif activation_type == "ReLU":
            self.activation = layers.ReLU()
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")
        
        self.layers_list.append(self.activation)
        
        # Create the sequential block
        self.block = tf.keras.Sequential(self.layers_list)

    def call(self, x):
        """
        Apply the convolutional block to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, length, inp_channels).

        Returns:
            tf.Tensor: Output tensor of shape (batch, length, out_channels).
        """
        return self.block(x)

    def get_config(self):
        """
        Return the config of the layer.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(Conv1dBlock, self).get_config()
        config.update({
            'layers_list': self.layers_list
        })
        return config

# Example Usage
if __name__ == "__main__":
    # Create sample inputs
    batch_size = 2
    length = 16
    channels = 8
    x = tf.random.normal((batch_size, length, channels))
    time_steps = tf.random.normal((batch_size, ))  # Example for SinusoidalPosEmb

    # Instantiate layers
    sinusoidal_emb = SinusoidalPosEmb(dim=16)
    downsample = Downsample1d(dim=8)
    upsample = Upsample1d(dim=8)
    conv_block = Conv1dBlock(
        inp_channels=8,
        out_channels=16,
        kernel_size=3,
        n_groups=4,
        activation_type="Mish"
    )

    # Apply layers
    emb = sinusoidal_emb(time_steps)
    down = downsample(x)
    up = upsample(down)
    conv = conv_block(x)

    print("Sinusoidal Embedding Shape:", emb.shape)
    print("Downsampled Shape:", down.shape)
    print("Upsampled Shape:", up.shape)
    print("Conv1dBlock Output Shape:", conv.shape)