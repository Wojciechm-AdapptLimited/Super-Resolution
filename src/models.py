from keras import layers, models
from src.blocks.layers import EncoderBlock,DecoderBlock

def create_autoencoder(img_shape,input_shape):
	model = models.Sequential()

	model.add(layers.InputLayer(img_shape+(3,)))
	model.add(layers.Resizing(*input_shape))
	model.add(EncoderBlock(8))
	model.add(EncoderBlock(16))
	model.add(EncoderBlock(32))
	# model.add(layers.GlobalAveragePooling2D())
	# model.add(layers.Dense(
	# model.add(layers.Reshape(target_shape=(45, 64, 128)))
	model.add(DecoderBlock(32))
	model.add(DecoderBlock(16))
	model.add(DecoderBlock(8))
	model.add(DecoderBlock(8))
	model.add(layers.Conv2D(8,kernel_size=(3, 3),strides=(1, 1), padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation('relu'))
	model.add(layers.Conv2D(3,kernel_size=(3, 3),strides=(1, 1), padding='same'))
	model.add(layers.Activation('sigmoid'))

	return model

def create_unet(img_shape, input_shape):
	inputs = layers.Input(shape=img_shape+(3,))

	resize = layers.Resizing(*input_shape)(inputs)

	conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(resize)
	#conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
	#conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
	#conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

	up4 = layers.UpSampling2D(size=(2, 2))(conv3)
	up4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
	#up4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up4)
	merged4 = layers.Concatenate(axis=3)([conv2, up4])

	up5 = layers.UpSampling2D(size=(2, 2))(merged4)
	up5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up5)
	#up5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up5)
	merged5 = layers.Concatenate(axis=3)([conv1, up5])

	up6 = layers.UpSampling2D(size=(2, 2))(merged5)
	output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up6)

	model = models.Model(inputs=inputs, outputs=output)

	return model

def create_srcnn(img_shape, input_shape):
	inputs = layers.Input(shape=img_shape+(3,))

	resize = layers.Resizing(*input_shape)(inputs)

	conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(resize)
	conv2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(conv1)
	conv3 = layers.Conv2D(3, (5, 5), activation='relu', padding='same')(conv2)

	return models.Model(inputs=inputs, outputs=conv3)
