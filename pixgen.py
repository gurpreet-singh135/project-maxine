import tensorflow as tf
import numpy as np


def downsample(filters, size, input_shape, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


class MyConv2Dtranspose(tf.keras.layers.Layer):
  def __init__(self, out_channels, out_shape, kernel_size, stride, padding, initializer, use_bias=False):
    super(MyConv2Dtranspose, self).__init__()
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.initializer = initializer
    self.use_bias = use_bias
    self.out_shape = out_shape

  def build(self, input_shape):
    self.filters = self.add_weight("filters",
                                   shape=[input_shape[2], input_shape[3], self.out_channels, input_shape[4]],
                                   initializer=self.initializer)
    self.frames = input_shape[1]
    self.in_shape = input_shape

  # @tf.function(experimental_relax_shapes=True)
  def call(self, input):
    frames = self.frames
    shape = tf.shape(input)
    in_shape = shape[-3:]
    batch_shape = shape[:-3]
    # x = tf.reshape(input,[-1] + in_shape.as_list())
    x = tf.reshape(input,tf.concat(([-1],in_shape),axis = -1))
    out1 = tf.nn.conv2d_transpose(x, self.filters, self.out_shape, self.stride, self.padding)
    out_shape = tf.shape(out1)[-3:]
    out = tf.reshape(out1,tf.concat((batch_shape,out_shape),axis = -1))
    # i = tf.constant(1)
    # # for i in range(1, frames):
    # condition = lambda i,out : i < frames
    # body = lambda i,out : (i+1 , tf.concat([out, tf.expand_dims(tf.nn.conv2d_transpose(input[:, i, :, :, :], self.filters, self.out_shape, self.stride, self.padding),axis=1)], axis=1))
    # i,out = tf.while_loop(condition,body,[i,out], shape_invariants=[i.get_shape(),tf.TensorShape([self.in_shape[0],None,out.shape[2],out.shape[3],out.shape[4]])])
    return out

def upsample(filters, size, out_shape, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(MyConv2Dtranspose(out_channels=filters,out_shape=out_shape,stride=2,kernel_size=3,padding='SAME',initializer=initializer))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def PixGenerator(input_shape=(10, 256, 256, 3), out_channels=3):
    inputs = tf.keras.layers.Input(shape=input_shape)
    input_shape = np.asarray((256, 256, 3))

    down_stack = [
        downsample(64, 4, (256, 256, 3), apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, (128, 128, 64)),  # (bs, 64, 64, 128)
        downsample(256, 4, (64, 64, 128)),  # (bs, 32, 32, 256)
        downsample(512, 4, (32, 32, 256)),  # (bs, 16, 16, 512)
        downsample(512, 4, (16, 16, 512)),  # (bs, 8, 8, 512)
        downsample(512, 4, (8, 8, 512)),  # (bs, 4, 4, 512)
        downsample(512, 4, (4, 4, 512)),  # (bs, 2, 2, 512)
        downsample(512, 4, (2, 2, 512)),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, (2, 2,), apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, (4, 4,), apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, (8, 8,), apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, (16, 16,)),  # (bs, 16, 16, 1024)
        upsample(256, 4, (32, 32,)),  # (bs, 32, 32, 512)
        upsample(128, 4, (64, 64,)),  # (bs, 64, 64, 256)
        upsample(64, 4, (128, 128,)),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = MyConv2Dtranspose(out_channels=out_channels,out_shape=(256,256),
                             kernel_size=3,
                             stride=2,
                             padding='SAME',
                             initializer=initializer)# (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # LSTM layer
    lstm = tf.keras.layers.LSTM(512, return_sequences=True)
    y = tf.squeeze(x, axis=2)
    y = tf.squeeze(y, axis=2)
    temp = lstm(y)
    temp = tf.expand_dims(temp, axis=2)
    temp = tf.expand_dims(temp, axis=2)
    x += temp

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

class PixGen(tf.keras.Model):
    def __init__(self, input_shape=(10,256,256,3), out_channels=3):
        super(PixGen, self).__init__(name='')
        down_stack = [
            downsample(64, 4, (256, 256, 3), apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample(128, 4, (128, 128, 64)),  # (bs, 64, 64, 128)
            downsample(256, 4, (64, 64, 128)),  # (bs, 32, 32, 256)
            downsample(512, 4, (32, 32, 256)),  # (bs, 16, 16, 512)
            downsample(512, 4, (16, 16, 512)),  # (bs, 8, 8, 512)
            downsample(512, 4, (8, 8, 512)),  # (bs, 4, 4, 512)
            downsample(512, 4, (4, 4, 512)),  # (bs, 2, 2, 512)
            downsample(512, 4, (2, 2, 512)),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, (2, 2,), apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample(512, 4, (4, 4,), apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample(512, 4, (8, 8,), apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample(512, 4, (16, 16,)),  # (bs, 16, 16, 1024)
            upsample(256, 4, (32, 32,)),  # (bs, 32, 32, 512)
            upsample(128, 4, (64, 64,)),  # (bs, 64, 64, 256)
            upsample(64, 4, (128, 128,)),  # (bs, 128, 128, 128)
        ]
        self.down_stack = down_stack
        self.up_stack = up_stack
        self.initializer = tf.random_normal_initializer(0., 0.02)
        last = MyConv2Dtranspose(out_channels=out_channels, out_shape=(256, 256),
                                 kernel_size=3,
                                 stride=2,
                                 padding='SAME',
                                 initializer=self.initializer)  # (bs, 256, 256, 3)
        self.last = last
    @tf.function
    def call(self,inputs):
        x = inputs

        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # LSTM layer
        lstm = tf.keras.layers.LSTM(512, return_sequences=True)
        y = tf.squeeze(x, axis=2)
        y = tf.squeeze(y, axis=2)
        temp = lstm(y)
        temp = tf.expand_dims(temp, axis=2)
        temp = tf.expand_dims(temp, axis=2)
        x += temp

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = self.last(x)
        return x
