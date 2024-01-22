import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D,UpSampling2D

class ConvolutionBlock(tf.keras.layers.Layer):
	def __init__(self, filters, sampling,kernel_size=(3, 3),activation='relu', strides=(1, 1),pool_size=(2,2), padding='same'):
		super(ConvolutionBlock, self).__init__()
		self.filters = filters
		self.kernel_size = kernel_size
		self.act_func = activation
		self.stride = strides
		self.pool_size = pool_size
		self.sampling = sampling
		self.padding = padding

	def build(self, input_shape):
		self.conv = Conv2D(self.filters, self.kernel_size, padding=self.padding)
		self.batch_norm = BatchNormalization()
		self.activation = Activation(self.act_func)
		self.sampling = self.sampling(self.pool_size)
		super(ConvolutionBlock, self).build(input_shape)

	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batch_norm(x)
		x = self.activation(x)
		x = self.sampling(x)
		return x

class EncoderBlock(ConvolutionBlock):
	def __init__(self, filters,kernel_size=(3, 3),activation='relu', strides=(1, 1),pool_size=(2,2), padding='same'):
		super(EncoderBlock,self).__init__(filters,MaxPooling2D,kernel_size,activation,strides,pool_size,padding)

class DecoderBlock(ConvolutionBlock):
	def __init__(self, filters,kernel_size=(3, 3),activation='relu', strides=(1, 1),pool_size=(2,2), padding='same'):
		super(DecoderBlock,self).__init__(filters,UpSampling2D,kernel_size,activation,strides,pool_size,padding)