
from tensorflow.keras.layers import Input, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Sequential, Model

def downsample_block(incoming_layer, nb_filters, kernel_size, batch_normalization):

	downsample_layer = Conv2D(nb_filters, kernel_size = kernel_size, strides = 2, padding = 'same')(incoming_layer)

	downsample_layer = LeakyReLU(0.2)(downsample_layer)

	if batch_normalization:
		downsample_layer = BatchNormalization(momentum = 0.8)(downsample_layer)

	return downsample_layer

def upsample_block(incoming_layer, skip_input_layer, nb_filters, kernel_size, dropout_rate):

	upsample_layer = UpSampling2D(size = 2)(incoming_layer)
	upsample_layer = Conv2D(nb_filters, kernel_size = kernel_size, strides = 1, padding = 'same', activation = 'relu')(upsample_layer)

	if dropout_rate:
		upsample_layer = Dropout(dropout_rate)(upsample_layer)

	upsample_layer = BatchNormalization(momentum = 0.8)(upsample_layer)

	upsample_layer = Concatenate()([upsample_layer, skip_input_layer])

	return upsample_layer

# U-Net Generator
def build_generator(img_shape, channels, nb_filters):

	input_layer = Input(shape = img_shape)

	# DOWNSAMPLING
	downsample_1 = downsample_block(input_layer, nb_filters, kernel_size = 4, batch_normalization = False)

	# Downsampling with batch normalization
	downsample_2 = downsample_block(downsample_1, nb_filters * 2, kernel_size = 4, batch_normalization = True)
	downsample_3 = downsample_block(downsample_2, nb_filters * 4, kernel_size = 4, batch_normalization = True)
	downsample_4 = downsample_block(downsample_3, nb_filters * 8, kernel_size = 4, batch_normalization = True)
	downsample_5 = downsample_block(downsample_4, nb_filters * 8, kernel_size = 4, batch_normalization = True)
	downsample_6 = downsample_block(downsample_5, nb_filters * 8, kernel_size = 4, batch_normalization = True)
	downsample_7 = downsample_block(downsample_6, nb_filters * 8, kernel_size = 4, batch_normalization = True)

	upsample_1 = upsample_block(downsample_7, downsample_6, nb_filters * 8, kernel_size = 4, dropout_rate = False)
	upsample_2 = upsample_block(upsample_1, downsample_5, nb_filters * 8, kernel_size = 4, dropout_rate = False)
	upsample_3 = upsample_block(upsample_2, downsample_4, nb_filters * 8, kernel_size = 4, dropout_rate = False)
	upsample_4 = upsample_block(upsample_3, downsample_3, nb_filters * 8, kernel_size = 4, dropout_rate = False)
	upsample_5 = upsample_block(upsample_4, downsample_2, nb_filters * 2, kernel_size = 4, dropout_rate = False)
	upsample_6 = upsample_block(upsample_5, downsample_1, nb_filters, kernel_size = 4, dropout_rate = False)

	upsample_7 = UpSampling2D(size = 2)(upsample_6)
	output_img = Conv2D(channels, kernel_size = 4, strides = 1, padding = 'same', activation = 'tanh')(upsample_7)

	return Model(input_layer, output_img)

