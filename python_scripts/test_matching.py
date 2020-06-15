"""Uses feature_vectors.txt file produced by process_database.py to match input images with an image in the
   database"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import sys
from os import path as p
from time import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import utils
from data import ImageDataRaw
from geo_utils import (check_coords_in_radius, coordinates_from_file,
                       get_center_from_coords)


def main(args):
    log_dir = args.log_dir
    # log_file_name = args.model.split()[-1] + '__' + args.feature_vectors_file
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.DEBUG, filemode='w')
    class_logger = config_logger('class_logger', '../testing_results/classes.log')
    overall_logger = config_logger('ovrl_logger', '../testing_results/results.log')

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            utils.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("image_batch_p:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load data
            dataset = utils.get_dataset(args.data_root)
            image_paths, _ = utils.get_image_paths_and_labels(dataset)
            images = ImageDataRaw(image_paths, sess, batch_size=1)
            nrof_images = len(image_paths)
            count = 0

            emb_array = None
            path_list = []
            coords_list = []
            line_count = 0
            with open(args.feature_vectors_file, "r") as file:
                for line in file:
                    # Calculate embedding
                    emb, coords, path = get_embedding_and_path(line)

                    if line_count % 100 == 0:
                        print(path)
                    line_count += 1

                    if emb_array is None:
                        emb_array = np.array(emb)
                    else:
                        emb_array = np.concatenate((emb_array, emb))
                    path_list.append(path)
                    coords_list.append(coords)

            emb_array = emb_array.reshape((-1, 512))
            print(emb_array.shape)

            class_true_count = 0
            class_all_count = 1
            last_class_name = ''
            duration = 0

            for i in range(nrof_images):
                #img_orig = cv2.imread(image_paths[i])
                #img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                #img = np.array(img)
                #img = img[np.newaxis, ...]
                img = images.batch()
                target_path = p.abspath(image_paths[i])
                target_path_short = p.split(target_path)[-1]
                target_class_name = get_querie_class_name(target_path)

                if last_class_name != target_class_name and last_class_name != '':
                    log_class_accuracy(last_class_name, class_true_count, class_all_count, class_logger)
                    class_all_count, class_true_count = 0, 0

                last_class_name = target_class_name

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                target_emb = sess.run(embeddings, feed_dict=feed_dict)

                # Calculate the area of search
                try:
                    target_gps_coords = coordinates_from_file(target_path)
                    center_characteristics = get_center_from_coords(target_gps_coords)
                    target_coords_are_none = False
                except:
                    target_coords_are_none = True

                img_file_list = []
                upper_bound = args.upper_bound
                start_time = time()

                for j in range(len(emb_array)):

                    # Check coords to be present and if so, whether they are in the target area
                    if (not args.use_coords) or (coords_list[j] is None) or target_coords_are_none or \
                            check_coords_in_radius(center_characteristics, coords_list[j]):
                        # Then calculate distance to the target
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb_array[j], target_emb[0]))))

                        # Insert a score with a path
                        img_file = ImageFile(path_list[j], dist)
                        img_file_list = insert_element(img_file, img_file_list, upper_bound=upper_bound)
                    else:
                        continue

                if top_n_accuracy(target_class_name, img_file_list, args.top_n, upper_bound):
                    print(target_class_name)
                    overall_logger.info(target_class_name + '/' + target_path_short)
                    count += 1
                    class_true_count += 1
                else:
                    print(target_class_name, list(map(str, img_file_list[:5])))
                    overall_logger.info(target_class_name + '/' + target_path_short + ' ' + str(list(map(str, img_file_list[:5]))))

                class_all_count += 1
                duration += time() - start_time

            log_class_accuracy(last_class_name, class_true_count, class_all_count, class_logger)
            print(count / nrof_images)
            print(duration / nrof_images)
            overall_logger.info('Total Accuracy: ' + str(count / nrof_images))


def config_logger(logger_name, file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    handler1 = logging.FileHandler(file)
    handler1.setLevel(level)
    logger.addHandler(handler1)
    return logger


def log_class_accuracy(last_class_name, class_true_count, class_all_count, logger):
    msg = 'Accuracy for {class_name}: {accuracy}'.format(
        class_name=last_class_name, accuracy=class_true_count / class_all_count)
    print(msg)
    logger.info(msg)


def top_1_accuracy(target_class_name, img_file_list):
    """Returns true if the first element of img_file_list is equal to target_class_name"""
    return get_querie_class_name(img_file_list[0].path) == target_class_name


def top_n_accuracy(target_class_name, img_file_list, n=4, upper_bound=25):
    top_n = set()
    i = 0
    while len(top_n) < n and i < upper_bound:
        class_name = get_querie_class_name(img_file_list[i].path)
        top_n.add(class_name)
        i += 1

    return target_class_name in top_n


def get_embedding_and_path(line):
    """Calculate embedding of an image"""
    line = line.strip()
    emb_dict = json.loads(line)
    path, emb = list(emb_dict.items())[0]
    coords = emb_dict["coordinates"]
    emb = np.array(emb)

    return emb, coords, path


def get_class_name(path):
    return p.split(p.dirname(path))[-1]


def get_querie_class_name(path):
    dir_name = get_class_name(path)
    return dir_name.split(r'__')[0]


def insert_element(element, sorted_list, upper_bound=25):
    """Inserts an element in a sorted list"""
    assert len(sorted_list) <= upper_bound

    if sorted_list == []:
        return [element]

    result_list = sorted_list
    for i, e in enumerate(sorted_list):
        if element <= e:
            left_part = sorted_list[:i]
            right_part = sorted_list[i:]
            left_part.append(element)
            left_part.extend(right_part)
            result_list = left_part

            if len(result_list) > upper_bound:
                result_list.pop()

            break

    if len(result_list) < upper_bound:
        result_list.append(element)

    return result_list


class ImageFile:
    def __init__(self, path, score):
        self.path = path
        self.score = score

    def __le__(self, other):
        return self.score <= other.score

    def __str__(self):
        return get_querie_class_name(self.path) + ':' + str(self.score)


def resize_and_merge(img_a, img_b):
    """Merges 2 images. Images should be passed as arrays"""
    shape_a, shape_b = img_a.shape, img_b.shape
    new_width = min(shape_a[0], shape_b[0], 1024)
    new_height = min(shape_a[1], shape_b[1], 512)

    resized_img_a = misc.imresize(img_a, (new_width, new_height), interp='bilinear')
    resized_img_b = misc.imresize(img_b, (new_width, new_height), interp='bilinear')

    assert resized_img_a.shape == resized_img_b.shape

    merged_img = np.concatenate((resized_img_a, resized_img_b), axis=1)

    return merged_img


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,
                        help='Path to data directory which needs to be forward passed through the network',
                        default='../datasets/queries_mixed/')
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf '
                             '(.pb) file',
                        default='../models/val84-with-resize')
    parser.add_argument('--feature_vectors_file', type=str,
                        help='Path to the file with feature vectors',
                        default='feature_vectors.txt')
    parser.add_argument('--log_dir', type=str,
                        help='Directory where to write event logs.',
                        default='../testing_results')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.',
                        default=256)
    parser.add_argument('--top_n', type=int,
                        help='Number of plausible classes of the input image',
                        default=4)
    parser.add_argument('--upper_bound', type=int,
                        help='How many images to try to match',
                        default=50)
    parser.add_argument('--use_coords', type=bool,
                        help='Whether to use GPS coordinates',
                        default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
