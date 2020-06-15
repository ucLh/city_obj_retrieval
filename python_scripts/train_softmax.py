"""Train a neural net for image retrieval task with TensorFlow using softmax cross entropy loss
"""

from __future__ import absolute_import, division, print_function

import argparse
import importlib
import math
import os.path
import random
import sys
import time
from datetime import datetime
from functools import partial

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
from aug.augmentations import random_black_patches
from data import LabeledImageData, LabeledImageDataRaw


def main(args):
    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    stat_file_name = os.path.join(log_dir, 'stat.h5')

    # Write arguments to a text file
    utils.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = utils.get_dataset(args.data_dir)
    # print(dataset[1].image_paths)
    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename),
                                 args.filter_percentile, args.filter_min_nrof_images_per_class)

    if args.validation_set_split_ratio > 0.0:
        train_set, val_set = utils.split_dataset(dataset, args.validation_set_split_ratio,
                                                   args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get a list of image paths and their labels
        image_list, label_list = utils.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The training set should not be empty'

        val_image_list, val_label_list = utils.get_image_paths_and_labels(val_set)

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')

        image_batch_plh = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image_batch_p')
        label_batch_plh = tf.placeholder(tf.int32, name='label_batch_p')

        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))

        print('Building training graph')

        # Build the inference graph
        # prelogits, _ = efficientnet_builder.build_model_base(image_batch_plh, 'efficientnet-b2', training=True)
        prelogits, _ = network.inference(image_batch_plh, args.keep_probability, image_size,
                                       phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # Add center loss
        prelogits_center_loss, _ = utils.center_loss(prelogits, label_batch_plh, args.center_loss_alfa, nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch_plh, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch_plh, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Separate facenet variables from smaug's ones
        facenet_global_vars = tf.global_variables()

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = utils.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, facenet_global_vars, args.log_histograms)

        # Create a saver
        facenet_saver_vars = tf.trainable_variables()
        facenet_saver_vars.append(global_step)
        saver = tf.train.Saver(facenet_saver_vars, max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Create normal pipeline
        dataset_train = LabeledImageData(image_list, label_list, sess, batch_size=args.batch_size, shuffle=True,
                                         use_flip=True, use_black_patches=True, use_crop=True)
        dataset_val = LabeledImageDataRaw(val_image_list, val_label_list, sess, batch_size=args.val_batch_size,
                                          shuffle=False)

        # Start running operations on the Graph. Change to tf.compat in newer versions of tf.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                ckpt_dir_or_file = tf.train.latest_checkpoint(pretrained_model)
                saver.restore(sess, ckpt_dir_or_file)

            # Training and validation loop
            print('Running training')
            nrof_steps = args.max_nrof_epochs * args.epoch_size
            nrof_val_samples = int(math.ceil(
                args.max_nrof_epochs / args.validate_every_n_epochs))  # Validate every validate_every_n_epochs as well as in the last epoch
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((args.max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((args.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.max_nrof_epochs, 1000), np.float32),
                'smaug_alpha_loss': np.zeros((nrof_steps,), np.float32),
                'smaug_total_loss': np.zeros((nrof_steps,), np.float32)
            }
            global_step_ = sess.run(global_step)
            start_epoch = 1 + global_step_ // args.epoch_size
            batch_number = global_step_ % args.epoch_size
            biggest_acc = 0.0
            for epoch in range(start_epoch, args.max_nrof_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()
                cont = train(args, sess, epoch, batch_number,
                             learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                             image_batch_plh, label_batch_plh, global_step,
                             total_loss, train_op, summary_op, summary_writer, regularization_losses,
                             args.learning_rate_schedule_file,
                             stat, cross_entropy_mean, accuracy, learning_rate, prelogits, prelogits_center_loss,
                             prelogits_norm, args.prelogits_hist_max, dataset_train, )
                stat['time_train'][epoch - 1] = time.time() - t
                print("------------------Accuracy-----------------" + str(stat['val_accuracy']))
                if not cont:
                    break

                t = time.time()
                if len(val_image_list) > 0 and ((epoch - 1) % args.validate_every_n_epochs == args.validate_every_n_epochs - 1 or epoch == args.max_nrof_epochs):
                    validate(args, sess, epoch, val_label_list, phase_train_placeholder, batch_size_placeholder,
                             stat, total_loss, cross_entropy_mean, accuracy, args.validate_every_n_epochs,
                             image_batch_plh, label_batch_plh, dataset_val)
                stat['time_validate'][epoch - 1] = time.time() - t

                cur_val_acc = get_val_acc(epoch, stat, args.validate_every_n_epochs)

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch, args.save_every,
                                             cur_val_acc, biggest_acc, args.save_best)

                biggest_acc = update_biggest_acc(biggest_acc, cur_val_acc)

                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.items():
                        f.create_dataset(key, data=value)

    return model_dir


def get_val_acc(epoch, stat, validate_every_n_epochs):
    val_index = (epoch - 1) // validate_every_n_epochs
    cur_val_acc = stat['val_accuracy'][val_index]
    return cur_val_acc


def update_biggest_acc(biggest_acc, cur_val_acc):
    if biggest_acc < cur_val_acc:
        biggest_acc = cur_val_acc
    return biggest_acc


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del (filtered_dataset[i])

    return filtered_dataset


def train(args, sess, epoch, batch_number,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
          image_batch_plh, label_batch_plh, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, prelogits_norm,
          prelogits_hist_max, dataset_train,):

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()

        # Process a batch of data for facenet
        image_batch, label_batch = dataset_train.batch()

        # Compute loss and perform training step for facenet
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size,
                     label_batch_plh: label_batch, image_batch_plh: image_batch}
        # image_summary = tf.summary.image('image_batch', image_batch)

        tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]

        if batch_number % 2 == 0:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, \
            lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = \
                sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
            # summary_writer.add_summary(image_summary_, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(
                tensor_list, feed_dict=feed_dict)

        duration = time.time() - start_time
        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = cross_entropy_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
        np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f'
              '\tLr %2.5f\tCl %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_),
               accuracy_, lr_, center_loss_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True


def validate(args, sess, epoch, label_list, phase_train_placeholder, batch_size_placeholder,
             stat, loss, cross_entropy_mean, accuracy, validate_every_n_epochs,
             image_batch_plh, label_batch_plh, dataset_val):

    print('Running forward pass on validation set')

    nrof_batches = len(label_list) // args.val_batch_size
    nrof_images = nrof_batches * args.val_batch_size

    loss_array = np.zeros((nrof_batches,), np.float32)
    xent_array = np.zeros((nrof_batches,), np.float32)
    accuracy_array = np.zeros((nrof_batches,), np.float32)

    # Training loop
    start_time = time.time()
    for i in range(nrof_batches):
        img, label = dataset_val.batch()
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: args.val_batch_size,
                     image_batch_plh: img, label_batch_plh: label}
        loss_, cross_entropy_mean_, accuracy_ = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
        loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')

    duration = time.time() - start_time

    val_index = (epoch - 1) // validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(loss_array)
    stat['val_xent_loss'][val_index] = np.mean(xent_array)
    stat['val_accuracy'][val_index] = np.mean(accuracy_array)

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, epoch, save_every,
                                 cur_val_acc, biggest_acc, save_best):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    if (epoch % save_every == 0) or (cur_val_acc > biggest_acc and save_best):
        saver.save(sess, checkpoint_path, global_step=epoch, write_meta_graph=False)
        print('Actually saved variables')
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, epoch)


def resize_images(image_batch_plh, image_size, use_black_patches_plh, use_random_crop_plh, use_random_flip_plh):

    load_size = (image_size[0] + 30, image_size[1] + 30)
    with tf.variable_scope('Augment'):
        resize_for_train = partial(tf.image.resize_images, size=load_size)
        resize_for_val = partial(tf.image.resize_images, size=image_size)
        images = tf.cond(
            tf.logical_or(tf.logical_or(use_black_patches_plh, use_random_crop_plh), use_random_flip_plh),
            lambda: tf.map_fn(lambda img: resize_for_train(img), image_batch_plh),
            lambda: tf.map_fn(lambda img: resize_for_val(img), image_batch_plh))

        # normalization
        images = tf.map_fn(lambda image: (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image)),
                           images)

        # augmentations
        images = tf.cond(use_black_patches_plh, lambda: tf.map_fn(random_black_patches, images), lambda: images)
        random_crop_fn = partial(tf.random_crop, size=(image_size[0], image_size[1], 3,))
        images = tf.cond(use_random_crop_plh, lambda: tf.map_fn(random_crop_fn, images), lambda: images)
        images = tf.cond(use_random_flip_plh, lambda: tf.map_fn(tf.image.random_flip_left_right, images),
                         lambda: images)

        # pylint: disable=no-member
        # image = tf.image.resize_images(image, image_size)

        image_batch_resized = tf.map_fn(lambda image: image * 2 - 1, images, name='image_batch_res')

    return image_batch_resized


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='../logs')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='../checkpoints')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--save_every', type=int,
                        help='Number of epochs to run.', default=20)
    parser.add_argument('--save_best', action='store_true',
                        help='Whether to save best current model during training')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='../datasets/current_train_rotated_resized/')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=40)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=2)
    parser.add_argument('--val_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=5)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=260)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=50)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_black_patches',
                        help='Adds random black patches to training images', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.4)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=5e-4)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
                        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0005)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=1)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='../data/learning_rate_schedule_classifier_casia.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
                        help='Number of epoch between validation', default=3)
    parser.add_argument('--validation_set_split_ratio', type=float,
                        help='The ratio of the total dataset to use for validation', default=0.05)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
                        help='Classes with fewer images will be removed from the validation set', default=0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
