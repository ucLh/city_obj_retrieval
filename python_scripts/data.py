"""Classes for managing datasets in tensorflow
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from aug.augmentations import random_black_patches


class ImageDataRaw:

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=1):
        self._sess = session
        self.image_paths = image_paths
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._drop_remainder = drop_remainder
        self._num_threads = num_threads
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._repeat = repeat
        self._img_batch = self._image_batch().make_one_shot_iterator().get_next()
        self._img_num = len(image_paths)

    def __len__(self):
        return self._img_num

    def batch(self):
        return self._sess.run(self._img_batch)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels, dct_method='INTEGER_ACCURATE')
        return img

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        return dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

    def _image_batch(self):
        dataset = self._map_dataset()

        if self._shuffle:
            dataset = dataset.shuffle(self._buffer_size)

        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = dataset.repeat(self._repeat).prefetch(2)

        return dataset


class ImageData(ImageDataRaw):

    def __init__(self,
                 image_paths,
                 session,
                 batch_size=10,
                 load_size=256,
                 use_black_patches=False,
                 use_crop=False,
                 crop_size=256,
                 use_flip=False,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=8,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=1):

        self._load_size = load_size
        self._use_black_patches = use_black_patches
        self._use_crop = use_crop
        self._crop_size = crop_size
        if use_crop:
            assert load_size >= crop_size
        self._use_flip = use_flip

        super().__init__(image_paths=image_paths,
                         session=session,
                         batch_size=batch_size,
                         num_channels=num_channels,
                         drop_remainder=drop_remainder,
                         num_threads=num_threads,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         repeat=repeat)

    def _parse_func(self, path):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels=self._num_channels, dct_method='INTEGER_ACCURATE')
        if self._use_flip:
            img = tf.image.random_flip_left_right(img)

        # img = tf.compat.v2.image.resize(img, [self._load_size, self._load_size])
        img = tf.image.resize_images(img, [self._load_size, self._load_size])
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))

        if self._use_black_patches:
            img = random_black_patches(img)

        if self._use_crop:
            img = tf.random_crop(img, [self._crop_size, self._crop_size, self._num_channels])
        else:
            img = tf.image.resize_images(img, [self._crop_size, self._crop_size])
        img = img * 2 - 1
        return img


class LabeledImageDataRaw(ImageDataRaw):

    def __init__(self,
                 image_paths,
                 labels,
                 session,
                 batch_size=10,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=False,
                 buffer_size=2048,
                 repeat=-1):
        self.labels = labels

        super().__init__(image_paths=image_paths,
                         session=session,
                         batch_size=batch_size,
                         num_channels=num_channels,
                         drop_remainder=drop_remainder,
                         num_threads=num_threads,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         repeat=repeat)

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        dataset = dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

        return dataset.zip((dataset, labels))


class LabeledImageData(ImageData):

    def __init__(self,
                 image_paths,
                 labels,
                 session,
                 batch_size=1,
                 load_size=286,
                 use_black_patches=False,
                 use_crop=True,
                 crop_size=256,
                 use_flip=True,
                 num_channels=3,
                 drop_remainder=True,
                 num_threads=16,
                 shuffle=True,
                 buffer_size=2048,
                 repeat=-1):

        self.labels = labels

        super().__init__(image_paths=image_paths,
                         session=session,
                         batch_size=batch_size,
                         load_size=load_size,
                         use_black_patches=use_black_patches,
                         use_crop=use_crop,
                         crop_size=crop_size,
                         use_flip=use_flip,
                         num_channels=num_channels,
                         drop_remainder=drop_remainder,
                         num_threads=num_threads,
                         shuffle=shuffle,
                         buffer_size=buffer_size,
                         repeat=repeat)

    def _map_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        dataset = dataset.map(self._parse_func, num_parallel_calls=self._num_threads)

        return dataset.zip((dataset, labels))
