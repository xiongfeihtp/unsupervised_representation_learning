'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: test.py.py
@time: 2018/4/30 下午7:44
@desc: shanghaijiaotong university
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import json
import numpy as np
from numpy import linalg as la
import tensorflow as tf
import random
import configuration
import Alex_Rotnet

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "./data/data_tfrecords/val-?????-of-00001",
                       "File pattern of sharded TFRecord input files.")

tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("inference_dir", "./log_test", "Directory to write event logs.")
tf.flags.DEFINE_integer("num_inference_examples", 10000,
                        "Number of examples for inference.")
tf.flags.DEFINE_string("test_image", None, "image path for validation for image similarity")
tf.flags.DEFINE_integer("top_k", 10, "top k similarity image")
tf.logging.set_verbosity(tf.logging.INFO)


# 余弦相似度
def cosSimilar(inA, inB):
    inA = np.mat(inA)
    inB = np.mat(inB)
    num = float(inA * inB.T)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def inference_model(sess, model, global_step, summary_writer, summary_op, model_config):
    # Log model summaries on a single batch.
    # summary_str = sess.run(summary_op)
    # summary_writer.add_summary(summary_str, global_step)
    # Compute perplexity over the entire dataset.
    num_inference_batches = int(
        math.ceil(FLAGS.num_inference_examples / model.config.batch_size))
    start_time = time.time()
    value_to_image_id = {}
    for i in range(num_inference_batches):
        extraction_feature = sess.run(model.feature)
        #extraction_feature = np.reshape(extraction_feature, (model_config.batch_size, -1))
        if not i % 100:  # more efficient
            tf.logging.info("Computed probabilities_value for %d of %d batches.", i + 1,
                            num_inference_batches)
        for item in zip(extraction_feature, model.image_names.eval()):
            value_to_image_id[item[1]] = item[0].tolist()
    inference_time = time.time() - start_time
    # Write the Events file to the eval directory.
    # summary_writer.flush()
    tf.logging.info("Finished processing inference at global step %d.inference_time (%.2g sec)",
                    global_step, inference_time)
    # get the most similarity image
    query = FLAGS.test_image
    if query is None:
        query = random.choice(list(value_to_image_id.keys()))
    dist_dict = {}
    for id in value_to_image_id:
        if id is not query:
            dist_dict[id] = cosSimilar(value_to_image_id[id], value_to_image_id[query])
    sorted_dict = sorted(dist_dict.items(), key=lambda item: item[1], reverse=True)
    decoder = ImageDecoder()
    for i in range(FLAGS.top_k):
        with tf.gfile.FastGFile(sorted_dict[i][0], 'rb') as f:
            image = f.read()
        summary_image = decoder.decode_jpeg(image)
        tf.summary.image(str(sorted_dict[i][1]), tf.expand_dims(summary_image, axis=0))
    with tf.gfile.FastGFile(query, 'rb') as f:
        image = f.read()
    summary_image = decoder.decode_jpeg(image)
    tf.summary.image("query_image", tf.expand_dims(summary_image, axis=0))
    summary_op = tf.summary.merge_all()
    summaries = sess.run(summary_op)
    summary_writer.add_summary(summaries, global_step)
    summary_writer.flush()

def run_once(model, saver, summary_writer, summary_op, model_config):
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping inference. No checkpoint found in: %s",
                        FLAGS.checkpoint_dir)
        return
    with tf.Session() as sess:
        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_path), global_step)
        # start on cpu for threads runners
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Run evaluation on the latest checkpoint.
        try:
            inference_model(
                sess=sess,
                model=model,
                global_step=global_step,
                summary_writer=summary_writer,
                summary_op=summary_op,
                model_config=model_config
            )

        except Exception as e:  # pylint: disable=broad-except
            tf.logging.error("inference failed.")
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def run():
    inference_dir = FLAGS.inference_dir
    if not tf.gfile.IsDirectory(inference_dir):
        tf.logging.info("Creating inference directory: %s", inference_dir)
        tf.gfile.MakeDirs(inference_dir)
    g = tf.Graph()

    with g.as_default():
        # Build the model for evaluation.
        model_config = configuration.ModelConfig()
        model_config.input_file_pattern = FLAGS.input_file_pattern
        model = Alex_Rotnet.Alex_RotNet(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(inference_dir)
        # g.finalize()
        tf.logging.info("Starting inference at " + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, summary_writer, summary_op, model_config)
        tf.logging.info("Competed inference")

def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.inference_dir, "--inference_dir is required"
    run()


if __name__ == "__main__":
    tf.app.run()
