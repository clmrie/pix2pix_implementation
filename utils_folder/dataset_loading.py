
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def read_separate_imgs(path):

	img = plt.imread(path, format = 'RGB').astype(np.float)
	h, w, _ = img.shape

	half_w = int(w/2)

	left_half_img = img[:, :half_w, :] / 127.5 - 1
	right_half_img = img[:, half_w:, :] / 127.5 - 1

	return left_half_img, right_half_img

def get_imgs(imgs_path_list, img_size, is_testing):

	imgs_source = []
	imgs_target = []

	for img_path in imgs_path_list:

		img_source, img_target = read_separate_imgs(img_path)
		img_source = tf.image.resize(img_source, img_size).numpy()
		img_target = tf.image.resize(img_target, img_size).numpy()

		# Flip random images horizontally during training only
		if not is_testing and np.random.random() > 0.5:

			img_source = np.fliplr(img_source)
			img_target = np.fliplr(img_target)

		imgs_source.append(img_source)
		imgs_target.append(img_target)

	imgs_source = np.array(imgs_source)
	imgs_target = np.array(imgs_target)

	return imgs_source, imgs_target

def batch_generator(loading_path, batch_size, img_shape, is_testing):

	data_type = 'train' if not is_testing else 'val'
	paths = glob(os.path.join(loading_path, data_type, '*.jpg'))

	nb_batches = int(len(paths)/batch_size)

	for i in range(nb_batches):

		batch_paths = paths[i * batch_size: (i+1) * batch_size]

		imgs_source, imgs_target = get_imgs(batch_paths, img_shape, is_testing)

		yield imgs_source, imgs_target

