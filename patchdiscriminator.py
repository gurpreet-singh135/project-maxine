import tensorflow as tf

class PatchDiscriminator(tf.keras.Model):
    """Construct a PatchGAN discriminator
    Parameters:
        input_nc (int)  -- the number of channels in input images
        channels (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """
    def __init__(self, channels=64, n_layers = 3, norm_layer = tf.keras.layers.BatchNormalization() ):
        super(PatchDiscriminator,self).__init__(name='')
        ops = tf.keras.Sequential()
        kernel_size = 4
        padding = 'same'
        ops.add(tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, strides= 2, padding= padding, input_shape=(4,256,256,3)))
        ops.add(tf.keras.layers.LeakyReLU(0.2))
        nf_mult = 1    #  for increasing number of channels
        for i in range(1, n_layers):
            nf_mult = min(2**i, 8)
            ops.add(tf.keras.layers.Conv2D(filters=channels*nf_mult, kernel_size=kernel_size, strides=2, padding= padding))
            ops.add(tf.keras.layers.BatchNormalization())
            ops.add(tf.keras.layers.LeakyReLU(0.2))

        nf_mult = min(2 ** n_layers, 8)
        ops.add(tf.keras.layers.Conv2D(filters=channels * nf_mult, kernel_size=kernel_size, strides=1, padding=padding))
        ops.add(tf.keras.layers.BatchNormalization())
        ops.add(tf.keras.layers.LeakyReLU(0.2))

        ops.add(tf.keras.layers.Conv2D(1,kernel_size=kernel_size, strides=1, padding= padding))
        self.ops = ops

    def call(self, input):
        return self.ops(input)


