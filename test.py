
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

import argparse
import tensorflow as tf
import numpy as np
import scipy
import os
import sys
import math

from blendingnetwork import *




####################################
# Parameters and argument parsing #
###################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_path", type=str,
    help="relative directory for all data")
parser.add_argument("output_path", type=str,
    help="base-level output directory for results")

parser.add_argument("--model_path", type=str, default="model",
    help="folder containing the tf network snapshots")

# Network training parameters
parser.add_argument("--direct_regression", default=False, action="store_true",
    help="Directly regress the output image instead of predicting blend weights")
parser.add_argument("--no_textured_mesh", default=False, action="store_true",
    help="Disable the input layer from the textured mesh")
parser.add_argument("--num_input_mosaics", type=int, default=4, 
    help="Number of input mosaic layers to use")

args = parser.parse_args()

direct_regression = args.direct_regression
use_global_mesh   = not args.no_textured_mesh
num_input_layers  = args.num_input_mosaics
batch_size        = 1

print("================================================")
print("Running deep blending network in TEST mode.")
if direct_regression:
    print("Directly predicting the output image.")
else:
    print("Predicting blend weights.")
print("Inputs: " + str(num_input_layers) + str(" mosaics") + \
    (str(" and a render of the textured mesh.") if use_global_mesh else str(".")))
print("================================================")




##########################
# Data loader definition #
##########################

class TestDataLoader(object):
    """Test mode data loader"""
    def __init__(self, filenames_file, data_folder):
        f = open(filenames_file, 'r')
        line = f.readline()
        f.close()
        self.init_width     = int(line.split()[0])
        self.init_height    = int(line.split()[1])
        self.current_images = None
        self.data_folder    = None

        with tf.variable_scope("inputloader"):
            self.data_folder = tf.constant(data_folder)

            input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
            line_reader = tf.TextLineReader()
            _, line     = line_reader.read(input_queue)
            split_line  = tf.string_split([line]).values

            offset = 0
            current_width = tf.string_to_number(split_line[offset], tf.int32)

            offset += 1
            current_height = tf.string_to_number(split_line[offset], tf.int32)

            offset += 1
            frames_color = []

            # First populate the CNN inputs with data from the global mesh
            global_color = self.read_jpg(split_line[offset])
            global_color.set_shape([self.init_height, self.init_width, 3])

            if use_global_mesh:
                frames_color.append(global_color)

            offset += 1
            # Then incorporate information from each per-view mosaic
            for i in range(num_input_layers):
                colors = self.read_jpg(split_line[offset + i])
                colors.set_shape([self.init_height, self.init_width, 3])
                frames_color.append(colors)

            images = tf.concat(frames_color, axis=2)
            
            # In test mode, we only use 1 thread for dataloading --- this prevents race conditions and loads the images in sequential order.
            min_after_dequeue   = 16
            capacity            = min_after_dequeue + 4
            dataloader_threads  = 1
            self.current_images = tf.train.batch([images], batch_size, dataloader_threads, capacity)

    def read_jpg(self, image_path):
        image = tf.image.decode_jpeg(tf.read_file(tf.string_join([self.data_folder, image_path], "/")))
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image




#########################
# Main code starts here #
#########################

data_file = args.data_path + "/test.txt"
dataloader = TestDataLoader(data_file, args.data_path)

with tf.variable_scope("standard_inputs"):
    current_images = dataloader.current_images
    current_input  = tf.concat(current_images, 3)

with tf.variable_scope("model", reuse=False):
    current_out, current_img_list = \
        blending_network(current_input, current_images, num_input_layers, use_global_mesh, direct_regression, batch_size)

# Save intermediate models
saver = tf.train.Saver(max_to_keep=1)

# Creating a config to prevent GPU use at all
config = tf.ConfigProto()

# Start GPU memory small and allow to grow
config.gpu_options.allow_growth=True

sess        = tf.Session(config=config)
init_global = tf.global_variables_initializer()
init_local  = tf.local_variables_initializer()
sess.run(init_local)
sess.run(init_global)

img_path   = args.output_path
model_path = args.model_path
if not os.path.isdir(img_path):
    os.makedirs(img_path, 0o755)

# Load pretrained weights
last_checkpoint = tf.train.latest_checkpoint(model_path)
if last_checkpoint != None:
    test_variables = []
    for var in tf.global_variables():
        if var.name.startswith("model") and var.name.find("Adam") == -1:
            test_variables.append(var)
    test_restorer = tf.train.Saver(test_variables)
    test_restorer.restore(sess, last_checkpoint)
else:
    print("Could not load model weights from: " + model_path)
    sys.exit(0)

# Start the data loader threads
coordinator = tf.train.Coordinator()
threads     = tf.train.start_queue_runners(sess=sess, coord=coordinator)

with sess.as_default():
    try:
        test_file  = open(data_file, 'r')
        test_lines = test_file.readlines()
        test_file.close()

        for i, line in enumerate(test_lines):
            if coordinator.should_stop():
                break

            output    = sess.run([current_out])
            raw_image = np.clip(output[0] * 255, 0, 255).astype(np.uint8)
            raw_image = np.reshape(raw_image, (dataloader.init_height, dataloader.init_width, 3))

            chunks      = line.split(" ")
            image_index = chunks[-1].split("/")[1].split("_")[0]
            if not os.path.isdir(img_path):
                os.makedirs(img_path, 0o755)
            scipy.misc.imsave(img_path + "/" + image_index + ".jpg", raw_image)

            print("Test frame: %s" % image_index)
    except Exception as e:
        # Report exceptions to the coordinator.
        coordinator.request_stop(e)
    finally:
        coordinator.request_stop()
        coordinator.join(threads)
