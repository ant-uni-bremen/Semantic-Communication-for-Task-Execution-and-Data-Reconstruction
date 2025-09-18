import tensorflow as tf
from keras import layers

class PatchEmbedding(layers.Layer):
    def __init__(self, image_size=224, patch_size=16, embed_dim=512, channels=3, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size * channels
        self.projection = layers.Dense(embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]

        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Dynamically compute num_patches and reshape
        patch_dims = tf.shape(patches)
        num_patches = patch_dims[1] * patch_dims[2]
        patches = tf.reshape(patches, [batch_size, num_patches, self.patch_dim])

        # Project patches to embedding dimension
        return self.projection(patches)
    
    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'patch_dim':self.patch_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            patch_dim=config['patch_dim'],
            name=config['name']  # Ensure no KeyError
        )
