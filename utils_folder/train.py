
import os
import numpy as np

import matplotlib.pyplot as plt

from dataset_loading import batch_generator

def show_test_results(imgs_source, imgs_cond, fake_imgs, saving_path, generator, epoch):

	imgs_to_show = [imgs_source, imgs_cond, fake_imgs]
	imgs_labels = ['imgs_source', 'imgs_cond', 'fake_imgs']

	for imgs_idx in range(len(imgs_to_show)):
		first_img = imgs_to_show[imgs_idx][0]
		first_img = (first_img + 1)/2

		print(imgs_labels[imgs_idx])

		plt.imshow(first_img)
		plt.show()

	saving_path = os.path.join(saving_path, str(epoch))
	os.mkdir(saving_path)
	generator.save(os.path.join(saving_path, 'generator.h5'))

def train(generator, discriminator, gan, patch_gan_shape, epochs, path, batch_size, img_size, print_every_n_batches, g_losses, d_losses, saving_path):

	real_y = np.ones((batch_size,) + patch_gan_shape)
	fake_y = np.zeros((batch_size,) + patch_gan_shape)

	for epoch in range(epochs):

		for idx, (imgs_source, imgs_cond) in enumerate(batch_generator(loading_path = path,
																	   batch_size = batch_size,
																	   img_shape = [img_size, img_size],
																	   is_testing = False)):

			fake_imgs = generator.predict([imgs_cond])

			disc_loss_real = discriminator.train_on_batch([imgs_source, imgs_cond], real_y)
			disc_loss_fake = discriminator.train_on_batch([fake_imgs, imgs_cond], fake_y)

			disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

			gan_loss = gan.train_on_batch([imgs_source, imgs_cond], [real_y, imgs_source])

			g_losses.append(gan_loss)
			d_losses.append(disc_loss)

		if epoch % 5 == 0:
			show_test_results(imgs_source, imgs_cond, fake_imgs, saving_path, generator, epoch)


			