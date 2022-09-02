
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam

def build_gan_model(shape, generator, discriminator):

	source_img = Input(shape = shape)
	cond_img = Input(shape = shape)

	fake_img = generator(cond_img)

	discriminator.trainable = False

	output = discriminator([fake_img, cond_img])

	gan = Model(inputs = [source_img, cond_img], outputs = [output, fake_img])
	gan.compile(loss = ['mse', 'mae'],
				loss_weights = [1, 100],
				optimizer = Adam(0.0002, 0.5))

	return gan