import tensorflow as tf
from tensorflow import keras
import einops
from copy import deepcopy
import logging

from model.common.mlp_tf import MLP, ResidualMLP
from model.diffusion.modules_tf import SinusoidalPosEmb
from model.common.modules_tf import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)

class DiffusionMLP(keras.Model):
    """MLP model for diffusion policies without vision input."""

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        test_mode=False,
    ):
        """Initialize the DiffusionMLP model.
        
        Args:
            action_dim (int): Dimension of the action space.
            horizon_steps (int): Number of horizon steps.
            cond_dim (int): Dimension of the conditioning input.
            time_dim (int, optional): Dimension of the time embedding. Defaults to 16.
            mlp_dims (list, optional): Dimensions of the MLP layers. Defaults to [256, 256].
            cond_mlp_dims (list, optional): Dimensions of the conditioning MLP layers. Defaults to None.
            activation_type (str, optional): Type of activation function. Defaults to "Mish".
            out_activation_type (str, optional): Type of output activation function. Defaults to "Identity".
            use_layernorm (bool, optional): Whether to use layer normalization. Defaults to False.
            residual_style (bool, optional): Whether to use residual MLP. Defaults to False.
            test_mode (bool, optional): Whether the model is in test mode. Defaults to False.
        """
        super().__init__()
        output_dim = action_dim * horizon_steps
        self.test_mode = test_mode
        if test_mode:
            self.time_embedding = keras.Sequential([
                SinusoidalPosEmb(time_dim),
                keras.layers.Activation("mish"),
            ])
        else:
            self.time_embedding = keras.Sequential([
                SinusoidalPosEmb(time_dim),
                keras.layers.Dense(time_dim * 2),
                keras.layers.Activation("mish"),
                keras.layers.Dense(time_dim),
            ])
        
        if residual_style:
            model_class = ResidualMLP
        else:
            model_class = MLP
    
            
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type=out_activation_type,
                test_mode=test_mode,
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim
            
        self.mlp_mean = model_class(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
            test_mode=test_mode,
        )
        self.time_dim = time_dim

    def call(self, x, time, cond, training=True, **kwargs):
        """Forward pass for DiffusionMLP.

        Args:
            x (tf.Tensor): Input tensor of shape (B, Ta, Da).
            time (tf.Tensor or int): Diffusion step, shape (B,).
            cond (dict): Conditioning dictionary with key 'state'.
            training (bool, optional): Training mode flag. Defaults to False.

        Returns:
            tf.Tensor: Output tensor of shape (B, Ta, Da).
        """
        B = tf.shape(x)[0]
        Ta = tf.shape(x)[1]
        Da = tf.shape(x)[2]
        
        # Flatten chunk
        x = tf.reshape(x, [B, -1])
        
        # Flatten history
        state = tf.reshape(cond["state"], [B, -1])
        
        # Obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state, training=training)
        
        # Append time and cond
        time = tf.reshape(time, [B, 1])
        time_emb = tf.reshape(self.time_embedding(time, training=training), [B, self.time_dim])
        x = tf.concat([x, time_emb, state], axis=-1)
        
        # MLP head
        out = self.mlp_mean(x, training=training)

        return tf.reshape(out, [B, Ta, Da])
    
class VisionDiffusionMLP(keras.Model):
    """MLP model for diffusion policies with vision input."""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        """Initialize the VisionDiffusionMLP model.
        
        Args:
            backbone (keras.Model): Backbone model for vision processing.
            action_dim (int): Dimension of the action space.
            horizon_steps (int): Number of horizon steps.
            cond_dim (int): Dimension of the conditioning input.
            img_cond_steps (int, optional): Number of image conditioning steps. Defaults to 1.
            time_dim (int, optional): Dimension of the time embedding. Defaults to 16.
            mlp_dims (list, optional): Dimensions of the MLP layers. Defaults to [256, 256].
            activation_type (str, optional): Type of activation function. Defaults to "Mish".
            out_activation_type (str, optional): Type of output activation function. Defaults to "Identity".
            use_layernorm (bool, optional): Whether to use layer normalization. Defaults to False.
            residual_style (bool, optional): Whether to use residual MLP. Defaults to False.
            spatial_emb (int, optional): Dimension for spatial embedding. Defaults to 0.
            visual_feature_dim (int, optional): Dimension of visual features. Defaults to 128.
            dropout (float, optional): Dropout rate. Defaults to 0.
            num_img (int, optional): Number of images. Defaults to 1.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        """
        super().__init__()
        
        # Vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                # Initialize multiple spatial embeddings if multiple images
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:
                # Initialize single spatial embedding or compression
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = keras.Sequential([
                keras.layers.Dense(visual_feature_dim),
                keras.layers.LayerNormalization(axis=1, epsilon=0.0001),
                keras.layers.Dropout(dropout),
                keras.layers.ReLU(),
            ])

        # Diffusion
        input_dim = time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
        
        self.time_embedding = keras.Sequential([
            SinusoidalPosEmb(time_dim),
            keras.layers.Dense(time_dim * 2),
            keras.layers.Activation("mish"),
            keras.layers.Dense(time_dim),
        ])
        
        if residual_style:
            model_class = ResidualMLP
        else:
            model_class = MLP
            
        self.mlp_mean = model_class(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def call(self, x, time, cond, training=False, **kwargs):
        """Forward pass for VisionDiffusionMLP.

        Args:
            x (tf.Tensor): Input tensor of shape (B, Ta, Da).
            time (tf.Tensor or int): Diffusion step, shape (B,).
            cond (dict): Conditioning dictionary with keys 'state' and 'rgb'.
            training (bool, optional): Training mode flag. Defaults to False.

        Returns:
            tf.Tensor: Output tensor of shape (B, Ta, Da).
        """
        B = tf.shape(x)[0]
        Ta = tf.shape(x)[1]
        Da = tf.shape(x)[2]
        
        # Flatten chunk
        x = tf.reshape(x, [B, -1])
        
        # Flatten history
        state = tf.reshape(cond["state"], [B, -1])
        
        # Take recent images
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        
        # Concatenate images in cond by channels
        if self.num_img > 1:
            T_rgb = tf.shape(rgb)[1]
            H = tf.shape(rgb)[3]
            W = tf.shape(rgb)[4]
            rgb = tf.reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        
        rgb = tf.cast(rgb, tf.float32)
        
        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                # Apply augmentation if enabled
                rgb1 = self.aug(rgb1, training=training)
                rgb2 = self.aug(rgb2, training=training)
            feat1 = self.backbone(rgb1, training=training)
            feat2 = self.backbone(rgb2, training=training)
            feat1 = self.compress1([feat1, state], training=training)
            feat2 = self.compress2([feat2, state], training=training)
            feat = tf.concat([feat1, feat2], axis=-1)
        else:
            if self.augment:
                # Apply augmentation if enabled
                rgb = self.aug(rgb, training=training)
            feat = self.backbone(rgb, training=training)
            
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress([feat, state], training=training)
            else:
                feat = tf.reshape(feat, [B, -1])
                feat = self.compress(feat, training=training)
                
        cond_encoded = tf.concat([feat, state], axis=-1)
        
        # Append time and cond
        time = tf.reshape(time, [B, 1])
        time_emb = tf.reshape(self.time_embedding(time, training=training), [B, self.time_dim])
        x = tf.concat([x, time_emb, cond_encoded], axis=-1)
        
        # MLP
        out = self.mlp_mean(x, training=training)
        return tf.reshape(out, [B, Ta, Da])