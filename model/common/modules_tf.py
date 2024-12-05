import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

class SpatialEmb(tf.keras.layers.Layer):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout=0.0):
        super(SpatialEmb, self).__init__()
        self.num_patch = num_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        self.proj_dim = proj_dim

        # Corrected: proj_in_dim should use patch_dim, not num_patch
        proj_in_dim = patch_dim + prop_dim

        self.input_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(proj_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
        ])

        self.weight = self.add_weight(
            shape=(1, num_patch, proj_dim),  # Changed: patch_dim to num_patch
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name='weight'
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, feat, prop, training=False):
        feat = tf.transpose(feat, perm=[0, 2, 1])

        if self.prop_dim > 0:
            repeated_prop = tf.expand_dims(prop, axis=1)
            repeated_prop = tf.tile(repeated_prop, [1, tf.shape(feat)[1], 1])
            feat = tf.concat([feat, repeated_prop], axis=-1)

        y = self.input_proj(feat)
        z = tf.reduce_sum(self.weight * y, axis=1)
        z = self.dropout(z, training=training)
        return z

# RandomShiftsAug Layer
class RandomShiftsAug(tf.keras.layers.Layer):
    def __init__(self, pad, **kwargs):
        super(RandomShiftsAug, self).__init__(**kwargs)
        self.pad = pad

    def call(self, x, training=False):
        if not training:
            return x

        # Assuming x is in shape (batch, height, width, channels)
        pad_size = self.pad
        x_padded = tf.pad(x, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], mode='REFLECT')

        # Generate random shifts
        batch_size = tf.shape(x)[0]
        max_shift = pad_size
        shifts = tf.random.uniform([batch_size, 2], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)

        def shift_image(args):
            img, shift = args
            shifted_img = tf.roll(img, shift, axis=[0,1])
            return shifted_img

        shifted = tf.map_fn(shift_image, (x_padded, shifts), dtype=tf.float32)

        # Crop back to original size
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        shifted_cropped = shifted[:, pad_size:pad_size + h, pad_size:pad_size + w, :]
        return shifted_cropped

    def get_config(self):
        config = super(RandomShiftsAug, self).get_config()
        config.update({
            'pad': self.pad
        })
        return config

# Image Loading Function
def load_image(url, size=(96, 96)):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = image.resize(size)
    image_np = np.array(image)
    image_np = image_np.astype(np.float32)
    return image_np

# Test Block
if __name__ == "__main__":
    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image_np = load_image(image_url, size=(96, 96))
    image_tensor = tf.expand_dims(image_np, axis=0)  # Shape: (1, H, W, C)

    # Define the augmentation layer
    aug = RandomShiftsAug(pad=4)

    # Apply augmentation
    image_aug = aug(image_tensor, training=True)
    image_aug = tf.squeeze(image_aug).numpy().astype(np.uint8)
    image_aug = Image.fromarray(image_aug.astype(np.uint8))
    image_aug.save('image_aug_test.png')