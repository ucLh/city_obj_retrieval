# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools

import tensorflow as tf

from aug import preprocessor_cache


def random_black_patches(image,
                         max_black_patches=10,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None,
                         preprocess_vars_cache=None):
    """Randomly adds some black patches to the image.

    This op adds up to max_black_patches square black patches of a fixed size
    to the image where size is specified via the size_to_image_ratio parameter.

    Args:
      image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
             with pixel values varying between [0, 1].
      max_black_patches: number of times that the function tries to add a
                         black box to the image.
      probability: at each try, what is the chance of adding a box.
      size_to_image_ratio: Determines the ratio of the size of the black patches
                           to the size of the image.
                           box_size = size_to_image_ratio *
                                      min(image_width, image_height)
      random_seed: random seed.
      preprocess_vars_cache: PreprocessorCache object that records previously
                             performed augmentations. Updated in-place. If this
                             function is called multiple times with the same
                             non-null cache, it will perform deterministically.

    Returns:
      image
    """

    def add_black_patch_to_image(image, idx):
        """Function for adding one patch to the image.

        Args:
          image: image
          idx: counter for number of patches that could have been added

        Returns:
          image with a randomly added black box
        """
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        box_size = tf.to_int32(
            tf.multiply(
                tf.minimum(tf.to_float(image_height), tf.to_float(image_width)),
                size_to_image_ratio))

        generator_func = functools.partial(tf.random_uniform, [], minval=0.0,
                                           maxval=(1.0 - size_to_image_ratio),
                                           seed=random_seed)
        normalized_y_min = _get_or_create_preprocess_rand_vars(
            generator_func,
            preprocessor_cache.PreprocessorCache.ADD_BLACK_PATCH,
            preprocess_vars_cache, key=str(idx) + 'y')
        normalized_x_min = _get_or_create_preprocess_rand_vars(
            generator_func,
            preprocessor_cache.PreprocessorCache.ADD_BLACK_PATCH,
            preprocess_vars_cache, key=str(idx) + 'x')

        y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
        x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
        black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
        mask = 1.0 - tf.image.pad_to_bounding_box(black_box, y_min, x_min,
                                                  image_height, image_width)
        image = tf.multiply(image, mask)
        return image

    with tf.name_scope('RandomBlackPatchInImage', values=[image]):
        for idx in range(max_black_patches):
            generator_func = functools.partial(tf.random_uniform, [],
                                               minval=0.0, maxval=1.0,
                                               dtype=tf.float32, seed=random_seed)
            random_prob = _get_or_create_preprocess_rand_vars(
                generator_func,
                preprocessor_cache.PreprocessorCache.BLACK_PATCHES,
                preprocess_vars_cache, key=idx)
            image = tf.cond(
                tf.greater(random_prob, probability), lambda: image,
                functools.partial(add_black_patch_to_image, image=image, idx=idx))
        return image


def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache,
                                        key=''):
    """Returns a tensor stored in preprocess_vars_cache or using generator_func.

    If the tensor was previously generated and appears in the PreprocessorCache,
    the previously generated tensor will be returned. Otherwise, a new tensor
    is generated using generator_func and stored in the cache.

    Args:
      generator_func: A 0-argument function that generates a tensor.
      function_id: identifier for the preprocessing function used.
      preprocess_vars_cache: PreprocessorCache object that records previously
                             performed augmentations. Updated in-place. If this
                             function is called multiple times with the same
                             non-null cache, it will perform deterministically.
      key: identifier for the variable stored.
    Returns:
      The generated tensor.
    """
    if preprocess_vars_cache is not None:
        var = preprocess_vars_cache.get(function_id, key)
        if var is None:
            var = generator_func()
            preprocess_vars_cache.update(function_id, key, var)
    else:
        var = generator_func()
    return var
