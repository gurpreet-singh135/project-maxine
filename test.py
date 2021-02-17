import tensorflow as tf

class MyDenseLayer(tf.keras.Model):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    self.frames = input_shape[1]

  def call(self, input):
    frames = self.frames
    for i in range(frames):
        out = tf.matmul(input[:,i,:],self.kernel)
    return out

class MyConv2Dtranspose(tf.keras.layers.Layer):
    def __init__(self, out_channels,out_shape, kernel_size, stride, padding, initializer, use_bias=False):
        super(MyConv2Dtranspose, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.use_bias = use_bias
        self.out_shape = out_shape

    def build(self, input_shape):
        self.filters = self.add_weight("filters", shape=[input_shape[2], input_shape[3], self.out_channels, input_shape[4]], initializer=self.initializer)
        self.frames = input_shape[1]
        self.in_shape = input_shape

    @tf.function
    def call(self, input):
        frames = self.frames
        out = tf.nn.conv2d_transpose(input[:, 0, :, :, :], self.filters, self.out_shape, self.stride, self.padding)
        out = tf.expand_dims(out, axis=1)
        for i in range(1,frames):
            out = tf.concat([out,tf.expand_dims(tf.nn.conv2d_transpose(input[:,i,:,:,:],self.filters,self.out_shape,self.stride,self.padding),axis=1)],axis = 1)
        return out

