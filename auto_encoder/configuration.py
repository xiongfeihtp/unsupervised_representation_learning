from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: configuration.py
@time: 2018/4/30 下午7:51
@desc: shanghaijiaotong university
'''
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Image-to-text model and training configurations."""
import tensorflow as tf
class ModelConfig(object):
  """Wrapper class for model hyperparameters."""
  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1
    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    self.label_feature_name = "image/image_label"
    self.filename_feature_name="image/image_name"
    # Batch size.
    self.batch_size = 32
    #image size
    self.image_height = 64
    self.image_width = 64
    self.scale = 1

    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 13321
    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 0.001
    self.learning_rate_decay_factor = 0.5
    self.num_epochs_per_decay = 8.0
    # If not None, clip gradients to this value.
    self.clip_gradients = 5.0
    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5

    self.num_steps = 60000
    self.checkpoint = 1000
    self.period = 10
    self.max_to_keep= 5

    self.run_id = '0'
    self.model_name = 'basic'

    self.load_path = None
    self.load_step = 0
    self.save_dir = './checkpoint'
    self.log_dir = './log'


