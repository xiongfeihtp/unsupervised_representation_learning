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

"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
def process_image(encoded_image,
                  is_training,
                  resize_height=224,
                  resize_width=224,
                  thread_id=0,
                  image_format="jpeg"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_summary("original_image", image)
  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  image_summary("resized_image", image)
  return image
