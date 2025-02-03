
import tensorflow as tf
from typing import Union
import einops
from copy import deepcopy

from model.common.mlp_tf import MLP, ResidualMLP  # TensorFlow versions
from model.common.modules_tf import SpatialEmb, RandomShiftsAug  # TensorFlow versions

class CriticObs(tf.keras.Model):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        test_mode=False,
        **kwargs,
    ):
        super(CriticObs, self).__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if test_mode:
            self.Q1 = tf.keras.layers.Dense(1)
        else:
            if residual_style:
                model = ResidualMLP
            else:
                model = MLP
            self.Q1 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def call(self, cond: Union[dict, tf.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = tf.shape(cond["state"])[0]
            state = tf.reshape(cond["state"], [B, -1])
        else:
            state = cond
        q1 = self.Q1(state)
        return q1


class CriticQvalue(tf.keras.Model):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        test_mode=False,
        **kwargs,
    ):
        super(CriticQvalue, self).__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if test_mode:
            self.Q1 = tf.keras.layers.Dense(1)
        else:
            if residual_style:
                model = ResidualMLP
            else:
                model = MLP
            self.Q1 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def call(self, cond: Union[dict, tf.Tensor], action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        if isinstance(cond, dict):
            B = tf.shape(cond["state"])[0]
            state = tf.reshape(cond["state"], [B, -1])
        else:
            state = cond

        flatten_action = tf.reshape(action, [B, -1])
        x = tf.concat([state, flatten_action], axis=-1)
        
        q1 = self.Q1(x)
        return q1

class CriticObsAct(tf.keras.Model):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        double_q=True,
        test_mode=False,
        **kwargs,
    ):
        super(CriticObsAct, self).__init__()

        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if test_mode:
            self.Q1 = tf.keras.layers.Dense(1)
            if double_q:
                self.Q2 = tf.keras.layers.Dense(1)
        else:
            if residual_style:
                model = ResidualMLP
            else:
                model = MLP
            self.Q1 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )
            if double_q:
                self.Q2 = model(
                    mlp_dims,
                    activation_type=activation_type,
                    out_activation_type="Identity",
                    use_layernorm=use_layernorm,
                )

    def call(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        B = tf.shape(cond["state"])[0]
        state = tf.reshape(cond["state"], [B, -1])
        action = tf.reshape(action, [B, -1])
        x = tf.concat([state, action], axis=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return tf.squeeze(q1, axis=1), tf.squeeze(q2, axis=1)
        else:
            q1 = self.Q1(x)
            return tf.squeeze(q1, axis=1)

class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super(ViTCritic, self).__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def call(
        self,
        cond: dict,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        no_augment: whether to skip augmentation
        """
        B, T_rgb, C, H, W = tf.shape(cond["rgb"])
        state = tf.reshape(cond["state"], [B, -1])
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        if self.num_img > 1:
            rgb = tf.reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, 'b t n c h w -> b n (t c) h w')
        else:
            rgb = einops.rearrange(rgb, 'b t c h w -> b (t c) h w')
        rgb = tf.cast(rgb, tf.float32)
        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1(feat1, state)
            feat2 = self.compress2(feat2, state)
            feat = tf.concat([feat1, feat2], axis=-1)
        else:
            if self.augment and not no_augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)
            feat = self.compress(feat, state)
        feat = tf.concat([feat, state], axis=-1)
        return super().call(feat)