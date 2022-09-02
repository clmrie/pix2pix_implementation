
from tensorflow.keras.layers import Input, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential, Model

def discriminator_block(incoming_layer, nb_filters, kernel_size, batch_normalization):

	disc_layer = Conv2D(nb_filters, kernel_size = kernel_size, strides = 2, padding = 'same')(incoming_layer)

	disc_layer = LeakyReLU(alpha = 0.2)(disc_layer)

	if batch_normalization:

		disc_layer = BatchNormalization(momentum = 0.8)(disc_layer)

	return disc_layer


# Patch-GAN discriminator
def build_discriminator(img_shape, channels, nb_filters):

	input_img = Input(shape = img_shape)
	cond_img = Input(shape = img_shape)
	
	combined_input = Concatenate(axis = -1)([input_img, cond_img])
	
	disc_block_1 = discriminator_block(combined_input, nb_filters, kernel_size = 4, batch_normalization = False)
	disc_block_2 = discriminator_block(disc_block_1, nb_filters * 2, kernel_size = 4, batch_normalization = True)
	disc_block_3 = discriminator_block(disc_block_2, nb_filters * 4, kernel_size = 4, batch_normalization = True)
	disc_block_4 = discriminator_block(disc_block_3, nb_filters * 8, kernel_size = 4, batch_normalization = True)

	output = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(disc_block_4)

	model = Model([input_img, cond_img], output)
	
	model.compile(loss = 'mse',
				  optimizer = Adam(0.0002, 0.5),
				  metrics = ['accuracy'])

	return model



