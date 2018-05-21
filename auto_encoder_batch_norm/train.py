'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: train.py.py
@time: 2018/4/30 下午7:43
@desc: shanghaijiaotong university
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import configuration
import auto_encoder_model
from tensorflow.contrib import slim
from graph_handler import GraphHandler
from tqdm import tqdm
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "./data/data_tfrecords/train-?????-of-00001",
                       "File pattern of sharded TFRecord input files.")

tf.flags.DEFINE_string("train_dir", "./train",
                       "Directory for saving and loading model checkpoints.")

tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)
    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = auto_encoder_model.Auto_Encoder_Model(
            model_config, mode="train")
        model.build()
        graph_handler = GraphHandler(model_config, model)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
            graph_handler.initialize(sess)
            for _ in tqdm(range(1, model_config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op, output = sess.run([model.total_loss, model.train_op, model.outputs_])
                if global_step % model_config.period == 0:
                    summary_op = tf.summary.merge_all()
                    summaries = sess.run(summary_op)
                    graph_handler.writer.add_summary(summaries, global_step)
                    graph_handler.writer.flush()
                if global_step % model_config.checkpoint == 0:
                    filename = os.path.join(
                        model_config.save_dir, "{}_{}.ckpt".format(model_config.model_name, global_step))
                    graph_handler.save(sess, filename)
            coord.join(queue_runner)
if __name__ == "__main__":
    tf.app.run()
