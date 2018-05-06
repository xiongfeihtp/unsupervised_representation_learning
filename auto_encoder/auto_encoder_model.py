'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: auto_encoder_model.py
@time: 2018/4/30 下午7:51
@desc: shanghaijiaotong university
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import image_processing
from ops import inputs as input_ops
from tensorflow.contrib import slim

class Auto_Encoder_Model(object):
    def __init__(self, config, mode):
        assert mode in ["train", "inference"]
        self.config = config
        self.mode = mode
        # Reader for the input data.
        self.reader = tf.TFRecordReader()
        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None
        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None
        # Global step Tensor.
        self.global_step = None
        self.scale = self.config.scale

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions.

        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              height=self.config.image_height,
                                              width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        # Prefetch serialized SequenceExample protos.
        input_queue = input_ops.prefetch_input_data(
            self.reader,
            self.config.input_file_pattern,
            is_training=self.is_training(),
            batch_size=self.config.batch_size,
            values_per_shard=self.config.values_per_input_shard,
            # approximate values nums for all shard
            input_queue_capacity_factor=self.config.input_queue_capacity_factor,
            # queue_capacity_factor for shards
            num_reader_threads=self.config.num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different color distortions.
        assert self.config.num_preprocess_threads % 2 == 0
        images_and_label = []
        for thread_id in range(self.config.num_preprocess_threads):
            # thread
            serialized_sequence_example = input_queue.dequeue()
            encoded_image, image_label, image_name = input_ops.parse_sequence_example(
                serialized_sequence_example,
                image_feature=self.config.image_feature_name,
                label_feature=self.config.label_feature_name,
                filename_feature=self.config.filename_feature_name)
            # preprocessing, for different thread_id use different distortion function
            image = self.process_image(encoded_image, thread_id=thread_id)

            images_and_label.append([image, image_name])
            # mutil threads preprocessing the image

        queue_capacity = (2 * self.config.num_preprocess_threads *
                          self.config.batch_size)
        images, image_names = tf.train.batch_join(
            images_and_label,
            batch_size=self.config.batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch")

        self.images = images  # batch_size, 224, 224 , 3
        #self.images = tf.cast(self.images, tf.float32)
        tf.summary.image("raw_images", self.images)
        self.images = tf.cast(self.images, tf.float32) / 255.
        if self.mode == "train":
            #with tf.variable_scope('noised'):
            #    self.inputs = tf.add(self.images, self.scale * tf.random_normal(shape=self.images.shape))
            self.inputs = self.images
            tf.summary.image("noised_images", self.images)
        elif self.mode == "inference":
            self.inputs = self.images
        self.image_names = image_names


    def encoder(self):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
            with tf.variable_scope('encoder'):
                conv1 = slim.conv2d(self.inputs, 64, [3, 3], scope='encoder_conv1')
                conv1 = slim.max_pool2d(conv1, [2, 2], stride=2, scope='encoder_maxpool1', padding ='SAME')  # [32, 32, 64]

                conv2 = slim.conv2d(conv1, 64, [3, 3], scope='encoder_conv2')
                conv2 = slim.max_pool2d(conv2, [2, 2], stride=2, scope='encoder_maxpool2', padding = 'SAME')  # [16, 16, 64]

                conv3 = slim.conv2d(conv2, 32, [3, 3], scope='encoder_conv3')
                self.conv3 = slim.max_pool2d(conv3, [2, 2], stride=2, scope='encoder_maxpool3', padding = 'SAME')  # [8, 8, 32]

    def decoder(self):
        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], padding='SAME', activation_fn=tf.nn.relu):
            with tf.variable_scope('decoder'):
                conv4 = tf.image.resize_nearest_neighbor(self.conv3, (16, 16))
                conv4 = slim.conv2d(conv4, 32, scope='decoder_conv1')

                conv5 = tf.image.resize_nearest_neighbor(conv4, (32, 32))
                conv5 = slim.conv2d(conv5, 64, scope='decoder_conv2')

                conv6 = tf.image.resize_nearest_neighbor(conv5, (64, 64))
                self.conv6 = slim.conv2d(conv6, 64, scope='decoder_conv3')
    def build_model(self):
        logits_ = slim.conv2d(self.conv6, 3, [3, 3], padding='SAME', activation_fn= None)
        self.outputs_ = tf.nn.sigmoid(logits_, name='output_')
        tf.summary.image("recontructed_images", self.outputs_ * 255.)
        loss = (self.outputs_- self.images) ** 2
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.images, logits=logits_)
        cost = tf.reduce_mean(loss)
        tf.losses.add_loss(cost)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("losses/total_loss", total_loss)
        self.total_loss = total_loss
    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step
    def build_train_op(self):
        if self.is_training():
            # Set up the learning rate.
            learning_rate = tf.constant(self.config.initial_learning_rate)
            if self.config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (self.config.num_examples_per_epoch /
                                         self.config.batch_size)
                decay_steps = int(num_batches_per_epoch *
                                  self.config.num_epochs_per_decay)

                learning_rate_decay_factor = self.config.learning_rate_decay_factor
                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=learning_rate_decay_factor,
                        staircase=True)
                self.lr = _learning_rate_decay_fn(learning_rate, self.global_step)
            tf.summary.scalar('learning_rate', self.lr)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.total_loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, self.config.clip_gradients)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
    def summary(self):
        self.summary_op= tf.summary.merge_all()
    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.encoder()
        self.decoder()
        self.build_model()
        self.setup_global_step()
        self.build_train_op()
        self.summary()
