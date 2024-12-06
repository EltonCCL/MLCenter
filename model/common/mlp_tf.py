"""
Implementation of Multi-Layer Perceptron (MLP) models in TensorFlow.

This module provides MLP and Residual MLP architectures with various configurations such as
activation functions, layer normalization, and dropout.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import logging

activation_dict = {
    "ReLU": layers.ReLU(),
    "ELU": layers.ELU(),
    "GELU": layers.Activation(tf.keras.activations.gelu),
    "Tanh": layers.Activation("tanh"),
    "Mish": layers.Activation("mish"),  # TensorFlow 2.2+ supports Mish
    "Identity": layers.Lambda(lambda x: x),
    "Softplus": layers.Activation("softplus"),
}


class MLP(tf.keras.Model):
    """
    Multi-Layer Perceptron (MLP) model with configurable layers, activations, normalization, and dropout.

    Args:
        dim_list (list): List of integers specifying the dimensions of each layer.
        append_dim (int, optional): Dimension of the data to append. Defaults to 0.
        append_layers (list, optional): Layers at which to append additional data. Defaults to None.
        activation_type (str, optional): Activation function type for hidden layers. Defaults to "Tanh".
        out_activation_type (str, optional): Activation function type for the output layer. Defaults to "Identity".
        use_layernorm (bool, optional): Whether to use Layer Normalization. Defaults to False.
        use_layernorm_final (bool, optional): Whether to apply Layer Normalization on the final layer. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.
        use_drop_final (bool, optional): Whether to apply dropout on the final layer. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        test_mode (bool, optional): Whether to set test mode for deterministic initialization. Defaults to False.
    """

    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
        test_mode=False,
    ):
        super(MLP, self).__init__()
        self.module_list = []
        self.append_layers = append_layers
        self.use_layernorm = use_layernorm
        self.use_layernorm_final = use_layernorm_final
        self.test_mode = test_mode
        num_layers = len(dim_list) - 1

        if test_mode:
            ones_initializer = tf.keras.initializers.Ones()
            kernel_initializer = ones_initializer
            bias_initializer = ones_initializer
        else:
            kernel_initializer = "glorot_uniform"
            bias_initializer = "zeros"

        for idx in range(num_layers):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and self.append_layers and idx in self.append_layers:
                i_dim += append_dim

            dense_layer = layers.Dense(
                o_dim,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                trainable=not test_mode,
            )

            self.module_list.append(dense_layer)

            if use_layernorm and (idx < num_layers - 1 or use_layernorm_final):
                layer_norm = layers.LayerNormalization(
                    gamma_initializer="ones" if test_mode else "ones",
                    beta_initializer="zeros",
                    trainable=not test_mode,
                )
                self.module_list.append(layer_norm)
            if dropout > 0 and (idx < num_layers - 1 or use_drop_final):
                self.module_list.append(layers.Dropout(dropout))

            act = (
                activation_dict[out_activation_type]
                if idx == num_layers - 1
                else activation_dict[activation_type]
            )
            self.module_list.append(act)

        if verbose:
            logging.info(self.module_list)

    def call(self, inputs, append=None):
        """
        Perform the forward pass of the MLP.

        Args:
            inputs (tf.Tensor): Input tensor.
            append (tf.Tensor, optional): Additional data to append at specified layers. Defaults to None.

        Returns:
            tf.Tensor: Output tensor after passing through the MLP.
        """
        x = inputs
        for layer_idx, layer in enumerate(self.module_list):
            if (
                append is not None
                and self.append_layers
                and layer_idx in self.append_layers
            ):
                x = tf.concat([x, append], axis=-1)
            x = layer(x)
        return x


class ResidualMLP(tf.keras.Model):
    """
    Residual Multi-Layer Perceptron (ResidualMLP) model with residual connections.

    Args:
        dim_list (list): List of integers specifying the dimensions of each layer.
        activation_type (str, optional): Activation function type. Defaults to "Mish".
        out_activation_type (str, optional): Activation function type for the output layer. Defaults to "Identity".
        use_layernorm (bool, optional): Whether to use Layer Normalization in residual blocks. Defaults to False.
        use_layernorm_final (bool, optional): Whether to apply Layer Normalization on the final layer. Defaults to False.
        dropout (float, optional): Dropout rate within residual blocks. Defaults to 0.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        test_mode=False,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert (
            num_hidden_layers % 2 == 0
        ), "Number of hidden layers must be even for residual connections."

        self.module_list = []  # Changed from self.layers to self.module_list
        self.module_list.append(layers.Dense(hidden_dim))
        for _ in range(1, num_hidden_layers, 2):
            self.module_list.append(
                TwoLayerPreActivationResNetLinear(
                    hidden_dim, activation_type, use_layernorm, dropout
                )
            )
        self.module_list.append(layers.Dense(dim_list[-1]))
        if use_layernorm_final:
            self.module_list.append(layers.LayerNormalization())
        self.module_list.append(activation_dict[out_activation_type])

    def call(self, inputs):
        """
        Perform the forward pass of the Residual MLP.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after passing through the Residual MLP.
        """
        x = inputs
        for layer in self.module_list:  # Changed from self.layers to self.module_list
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(tf.keras.layers.Layer):
    """
    A two-layer pre-activation Residual block for ResNet-like architectures.

    Args:
        hidden_dim (int): Dimension of the hidden layers.
        activation_type (str, optional): Activation function type. Defaults to "Mish".
        use_layernorm (bool, optional): Whether to use Layer Normalization. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.
    """

    def __init__(
        self, hidden_dim, activation_type="Mish", use_layernorm=False, dropout=0
    ):
        super(TwoLayerPreActivationResNetLinear, self).__init__()
        self.use_layernorm = use_layernorm
        self.l1 = layers.Dense(hidden_dim)
        self.l2 = layers.Dense(hidden_dim)
        self.act = activation_dict[activation_type]
        if self.use_layernorm:
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout not implemented; add if needed

    def call(self, inputs):
        """
        Perform the forward pass of the Residual block.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying residual connections.
        """
        x_input = inputs
        if self.use_layernorm:
            x = self.norm1(inputs)
        else:
            x = inputs
        x = self.act(self.l1(x))
        if self.use_layernorm:
            x = self.norm2(x)
        x = self.act(self.l2(x))
        return x + x_input
