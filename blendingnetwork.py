# Copyright 2019 Peter Hedman. All Rights Reserved.
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

######################################
# Building our Deep Blending network #
######################################

import tensorflow as tf

def blending_network(all_inputs, images, num_input_layers, use_global_mesh, direct_regression, batch_size):
    # We're using the NCHW data layout -- tensors should be concatenated 
    # along the 1st (channels) dimension.
    concat_dimension = 1

    # Helper functions for the convolutional layers
    def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu):
        c = tf.contrib.layers.conv2d(
            x, num_out_layers, kernel_size, stride, 'SAME', data_format='NCHW',
            activation_fn=activation_fn)
        return c
    def upsample_nn(x, num_in_layers):
        x_up = tf.reshape(
            tf.stack([x, x], axis=3),
            [tf.shape(x)[0], num_in_layers, 2 * tf.shape(x)[2], tf.shape(x)[3]])
        return tf.reshape(
            tf.concat([x_up[...,tf.newaxis], x_up[...,tf.newaxis]], axis=-1),
            [tf.shape(x)[0], num_in_layers, 2 * tf.shape(x)[2], 2 * tf.shape(x)[3]])
    
    def upconv(x, num_out_layers, kernel_size):
            upsample = upsample_nn(x, x.shape[1])
            conv1 = conv(upsample, num_out_layers, kernel_size, 1)
            return conv1

    def downconv(x, num_out_layers, kernel_size, stride = 2):
            conv1 = conv(x,     num_out_layers, kernel_size, 1)
            conv2 = conv(conv1, num_out_layers, kernel_size, stride)
            return conv2

    # Convert the inputs to NCHW
    all_inputs = tf.transpose(all_inputs, [0, 3, 1, 2])

    # The deep blending U-net
    conv1    = conv(all_inputs,  32, 3, 1)
    conv2    = downconv(conv1,   48, 3, 2)
    conv3    = downconv(conv2,   64, 3, 2)
    conv4    = downconv(conv3,   96, 3, 2)
    conv5    = downconv(conv4,  128, 3, 2)
    conv6    = upconv(conv5,     96, 3)
    concat1  = tf.concat([conv6, conv4], concat_dimension)
    conv7    = upconv(concat1,   64, 3)
    concat2  = tf.concat([conv7, conv3], concat_dimension)
    conv8    = upconv(concat2,   48, 3)
    concat3  = tf.concat([conv8, conv2], concat_dimension)
    conv9    = upconv(concat3,   32, 3)
    features = tf.concat([conv9, conv1], concat_dimension)

    # obtain the final output
    num_images = num_input_layers
    if use_global_mesh:
        num_images = num_images + 1

    img_list = tf.split(images, num_images, 3)
    if direct_regression:
        out_image = 0.5 * (1.0 + conv(features, 3, 3, 1, tf.nn.tanh))
    else:
        out = conv(features, num_images, 3, 1, None)
        softmax = tf.nn.softmax(out, 1)

        out = tf.reshape(
            all_inputs, [batch_size, num_images, 3, tf.shape(out)[2], tf.shape(out)[3]])
        out *= softmax[:,:,tf.newaxis]
        out_image = tf.reduce_sum(out, axis=1)

    # Convert back to NHCW
    out_image = tf.transpose(out_image, [0, 2, 3, 1])

    return out_image, img_list