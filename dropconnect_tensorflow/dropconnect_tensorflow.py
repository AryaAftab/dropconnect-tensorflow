import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import ReLU, BatchNormalization, Flatten, MaxPool2D, Input




#classes

class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1]]')


        super(DropConnectDense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.in_feature = input_shape[-2]
        
        super(DropConnectDense, self).build(input_shape)

    def call(self, inputs, train=False):
        if train:
            mask = tf.cast(tf.random.uniform((self.in_feature, self.units)) <= self.prob, tf.float32)
            kernel = tf.multiply(self.kernel, mask)
            output = tf.matmul(inputs, kernel)
        else:
            output = tf.matmul(inputs, self.kernel)
          
        if self.use_bias:
            output += self.bias
        return self.activation(output)
        
        
        
        
        
        
class DropConnectConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if not 0. <= self.prob < 1.:
            raise NameError('prob must be at range [0, 1]]')

        super(DropConnectConv2D, self).__init__(*args, **kwargs)

        if type(self.padding) is str:
            self.padding = self.padding.upper()

        
    def build(self, input_shape):
        self.in_channel = input_shape[-1]
        
        super(DropConnectConv2D, self).build(input_shape)

    def call(self, inputs, train=False):
        if train:
            mask = tf.cast(tf.random.uniform((self.kernel_size[0], self.kernel_size[1], self.self.in_channel, self.filters)) <= self.prob, tf.float32)
            kernel = tf.multiply(self.kernel, mask)
            output = tf.nn.conv2d(inputs,
                                  kernel,
                                  strides=self.strides,
                                  padding=self.padding,
                                  dilations=self.dilation_rate)
        else:
            output = tf.nn.conv2d(inputs,
                                  self.kernel,
                                  strides=self.strides,
                                  padding=self.padding,
                                  dilations=self.dilation_rate)
          
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return self.activation(output)
