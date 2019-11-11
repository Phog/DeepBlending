
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
import os
import sys
import math
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg

from bilinear_sampler import *
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

parser.add_argument("--training_file", type=str, default=None,
    help="full path to the training file")
parser.add_argument("--validation_file", type=str, default=None,
    help="full path to the validation file")

# Define the input and output folders
parser.add_argument("--log_path", type=str, default="log")
parser.add_argument("--model_path", type=str, default="model")

# Network training parameters
parser.add_argument("--loss_function", type=str, default="VGG_AND_L1",
    help="Image loss to be used for training (L1, VGG, or VGG_AND_L1)")
parser.add_argument("--direct_regression", default=False, action="store_true",
    help="Directly regress the output image instead of predicting blend weights")
parser.add_argument("--no_temporal_loss", default=False, action="store_true",
    help="Train the network without a temporal loss")
parser.add_argument("--no_textured_mesh", default=False, action="store_true",
    help="Disable the input layer from the textured mesh")
parser.add_argument("--num_input_mosaics", type=int, default=4, 
    help="Number of input mosaic layers to use")
parser.add_argument("--temporal_alpha", type=float, default=0.33, 
    help="Relative strength of the temporal loss")
parser.add_argument("--debug", default=False, action="store_true",
    help="Enable debug mode for training")
parser.add_argument("--num_batches", type=int, default=256000,
    help="Training duration (in terms of number of minibatches)")
parser.add_argument("--batch_size", type=int, default=8,
    help="Batch size to be used for training")
parser.add_argument("--crop", type=int, default=256,
    help="Crop size for data augmentation")

args = parser.parse_args()

image_loss        = args.loss_function
direct_regression = args.direct_regression
use_temporal_loss = not args.no_temporal_loss
use_global_mesh   = not args.no_textured_mesh
num_input_layers  = args.num_input_mosaics
debug_mode        = args.debug
temporal_alpha    = args.temporal_alpha
crop              = args.crop
num_batches       = args.num_batches
batch_size        = args.batch_size

print("================================================")
print("Running deep blending network in TRAINING mode.")
print("Loss function: " + image_loss + str("."))
if direct_regression:
    print("Directly predicting the output image.")
else:
    print("Predicting blend weights.")
print("Inputs: " + str(num_input_layers) + str(" mosaics") + \
    (str(" and a render of the textured mesh.") if use_global_mesh else str(".")))
print("Training " + (str("WITH") if use_temporal_loss else str("WITHOUT")) + str(" a temporal loss."))
if use_temporal_loss:
    print("Temporal alpha=" + str(temporal_alpha))
if debug_mode:
    print("Running in debug mode.")
print("Random crop size: " + str(crop) + "x" + str(crop))
print("Batch size: " + str(batch_size))
print("Training until " + str(num_batches) + " minibatches")
print("================================================")




##########################
# Data loader definition #
##########################

def augment_image(img, flip_vert, flip_horiz, rotate_img) :
    img = tf.cond(flip_vert > 0.5, lambda: tf.image.flip_up_down(img), lambda: img)
    img = tf.cond(flip_horiz > 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(rotate_img > 0.5, lambda: tf.image.rot90(img), lambda: img)
    return img

def augment_flow(img, flip_vert, flip_horiz, rotate_img):
    x_flow = tf.expand_dims(img[:,:,0], 2)
    y_flow = tf.expand_dims(img[:,:,1], 2)
    z_flow = tf.expand_dims(img[:,:,2], 2)

    # Vertical flipping
    y_flow = tf.cond(flip_vert > 0.5, lambda: y_flow * -1.0, lambda: y_flow)

    # Horizontal flipping
    x_flow = tf.cond(flip_horiz > 0.5, lambda: x_flow * -1.0, lambda: x_flow)

    # 90 degree rotation
    new_x_flow = tf.cond(rotate_img > 0.5, lambda: y_flow * -1.0, lambda: x_flow)
    new_y_flow = tf.cond(rotate_img > 0.5, lambda: x_flow, lambda: y_flow)

    return augment_image(tf.concat([new_x_flow, new_y_flow, z_flow], axis=2),
        flip_vert, flip_horiz, rotate_img)

class TrainingDataLoader(object):
    """Training mode data loader"""
    def __init__(self, filenames_file, data_folder):
        self.ref_images     = None
        self.current_images = None
        self.old_images     = None
        self.old_to_current = None
        self.init_height    = None
        self.init_width     = None
        self.data_folder    = None

        with tf.variable_scope("inputloader"):
            self.data_folder = tf.constant(data_folder)

            input_queue = tf.train.string_input_producer([filenames_file], shuffle = True)
            line_reader = tf.TextLineReader()
            _, line     = line_reader.read(input_queue)
            split_line  = tf.string_split([line]).values

            offset = 0
            current_width = tf.string_to_number(split_line[offset], tf.int32)

            offset += 1
            current_height = tf.string_to_number(split_line[offset], tf.int32)

            # Load the reference image
            offset += 1
            ref_img = self.read_jpg(split_line[offset])
            ref_img.set_shape([self.init_height, self.init_width, 3])

            # Augment the reference image
            flip_vert  = tf.random_uniform([], 0, 1)
            flip_horiz = tf.random_uniform([], 0, 1)
            rotate_img = tf.random_uniform([], 0, 1)

            crop_x = tf.random_uniform([], 0, current_width - crop, dtype=tf.int32)
            crop_y = tf.random_uniform([], 0, current_height - crop, dtype=tf.int32)

            asserts_ref = [
                tf.assert_greater_equal(current_width,  crop, message="Current width smaller than crop size"),
                tf.assert_greater_equal(current_height, crop, message="Current height smaller than crop size"),
                tf.assert_greater_equal(tf.shape(ref_img)[0], crop_y + crop, message="Reference height smaller than crop size"),
                tf.assert_greater_equal(tf.shape(ref_img)[1], crop_x + crop, message="Reference width smaller than crop size")]

            with tf.control_dependencies(asserts_ref):
                ref_img = tf.image.crop_to_bounding_box(ref_img, crop_y, crop_x, crop, crop)
                ref_img = augment_image(ref_img, flip_vert, flip_horiz, rotate_img)

            # Optionally: Load the optical flow between the two temporal frames
            offset  += 1
            flow_img = tf.zeros_like(ref_img)
            if use_temporal_loss:
                flow_img = self.read_flow(split_line[offset])

                # Also augment the current-to-previous optical flow
                asserts_flow = [
                    tf.assert_greater_equal(tf.shape(flow_img)[0], crop_y + crop, message="Temporal flow height smaller than crop size"),
                    tf.assert_greater_equal(tf.shape(flow_img)[1], crop_x + crop, message="Temporal flow width smaller than crop size")]
                with tf.control_dependencies(asserts_flow):
                    flow_img = tf.image.crop_to_bounding_box(flow_img, crop_y, crop_x, crop, crop)
                    flow_img = augment_flow(flow_img, flip_vert, flip_horiz, rotate_img)

            # Always load two temporal frames for path samples -- we'll ignore these
            # later during training if use_temporal_loss is False.
            num_path_samples = 2
            frames_color     = []
            offset          += 1

            for j in range(num_path_samples):
                # Our training data has 5 input layers per path sample:
                #  global_colors and local_layer_N_colors (where N=0...4)
                path_sample_offset = 5 * j

                # First populate the CNN inputs with data from the global mesh
                if use_global_mesh:
                    global_color = self.read_jpg(split_line[offset + path_sample_offset + 0])
                    global_color.set_shape([self.init_height, self.init_width, 3])

                    # Make sure to apply the same data augmentation to the textured global mesh layer
                    asserts_global = [
                        tf.assert_greater_equal(tf.shape(global_color)[0], crop_y + crop, message="Textured mesh height smaller than crop size"),
                        tf.assert_greater_equal(tf.shape(global_color)[1], crop_x + crop, message="Textured mesh width smaller than crop size")]
                    with tf.control_dependencies(asserts_global):
                        global_color = tf.image.crop_to_bounding_box(global_color, crop_y, crop_x, crop, crop)
                        global_color = augment_image(global_color, flip_vert, flip_horiz, rotate_img)

                    frames_color.append([global_color])
                else:
                    frames_color.append([])

                # Index where the image mosaic layers begin (+1 is for the textured mesh layer above)
                mosaic_layers_begin = offset + path_sample_offset + 1
                for i in range(num_input_layers):
                    colors = self.read_jpg(split_line[mosaic_layers_begin + i])
                    colors.set_shape([self.init_height, self.init_width, 3])

                    # Augment the image mosaics
                    asserts_local = [
                        tf.assert_greater_equal(tf.shape(colors)[0], crop_y + crop, message="Mosaic height smaller than crop size"),
                        tf.assert_greater_equal(tf.shape(colors)[1], crop_x + crop, message="Mosaic width smaller than crop size")]
                    with tf.control_dependencies(asserts_local):
                        colors = tf.image.crop_to_bounding_box(colors, crop_y, crop_x, crop, crop)
                        colors = augment_image(colors, flip_vert, flip_horiz, rotate_img)
                    frames_color[-1].append(colors)

            images_current  = tf.concat(frames_color[0], axis=2)
            images_previous = tf.concat(frames_color[1], axis=2)

            dataloader_threads = 8
            min_after_dequeue  = 16
            capacity           = min_after_dequeue + 4 * batch_size
            if use_temporal_loss:
                self.ref_images, self.current_images, self.old_images, self.old_to_current = \
                    tf.train.shuffle_batch([ref_img, images_current, images_previous, flow_img], batch_size, capacity, min_after_dequeue, dataloader_threads)
            else:
                self.ref_images, self.current_images, self.old_images = \
                    tf.train.shuffle_batch([ref_img, images_current, images_previous], batch_size, capacity, min_after_dequeue, dataloader_threads)

    def read_flow(self, image_path):
        image = tf.image.decode_png(tf.read_file(tf.string_join([self.data_folder, image_path], "/")), channels=3, dtype=tf.uint16)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([self.init_height, self.init_width, 3])
        image = 512.0 * (image * 2.0 - 1.0)
        return image

    def read_jpg(self, image_path):
        image = tf.image.decode_jpeg(tf.read_file(tf.string_join([self.data_folder, image_path], "/")))
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image




############################################
# Helper functions for our training losses #
############################################

def compute_standard_error(output, reference):
    l1_image = tf.abs(output - reference)
    return l1_image

vgg_mean = tf.reshape(tf.constant([123.68, 116.78, 103.94]), [1, 1, 3])
def vgg_16(inputs, scope='vgg_16'):
  """Computes deep image features as the first two maxpooling layers of a VGG16 network"""
  with tf.variable_scope('vgg_16', 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net_a = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net_b = slim.max_pool2d(net_a, [2, 2], scope='pool1')
      net_c = slim.repeat(net_b, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      return net_a, net_c

def compute_vgg_error(output, reference, layer):
    scaled_output = output * 255 - vgg_mean
    scaled_reference = reference * 255 - vgg_mean
    with slim.arg_scope(vgg.vgg_arg_scope()):
        output_a, output_b = vgg_16(scaled_output)
        reference_a, reference_b = vgg_16(scaled_reference)
    if layer == 0:
        return tf.abs(output_a - reference_a)
    return tf.abs(output_b - reference_b)

if image_loss == 'L1': # Standard L1
    def compute_error(output, reference, kernel_size):
        return compute_standard_error(output, reference)

    def compute_loss(output, reference, kernel_size):
        return tf.reduce_mean(compute_standard_error(output, reference))
elif image_loss == 'VGG': # Use a VGG loss
    def compute_error(output, reference, kernel_size):
        return compute_vgg_error(output, reference, 0)

    def compute_loss(output, reference, kernel_size):
        return tf.reduce_mean(compute_vgg_error(output, reference, 0)) + \
            tf.reduce_mean(compute_vgg_error(output, reference, 1))
elif image_loss == 'VGG_AND_L1': # Mix L1 and VGG to perserve high frequency content
    def compute_error(output, reference, kernel_size):
        return 255.0 * compute_standard_error(output, reference)

    def compute_loss(output, reference, kernel_size):
        return tf.reduce_mean(compute_vgg_error(output, reference, 0)) + \
            tf.reduce_mean(compute_vgg_error(output, reference, 1)) + \
            255.0 * tf.reduce_mean(compute_standard_error(output, reference))
else:
    print("Unexpected image loss: " + image_loss)
    sys.exit(0)




#########################
# Main code starts here #
#########################

data_file  = args.training_file
dataloader = TrainingDataLoader(data_file, args.data_path)

dataloader_validation = None
if args.validation_file is not None:
    validation_file       = args.validation_file
    dataloader_validation = TrainingDataLoader(validation_file, args.data_path)

# Collect inputs from the data loaders
with tf.variable_scope("standard_inputs"):
    ref_image = dataloader.ref_images
    current_images = dataloader.current_images
    old_images = dataloader.old_images
    current_input = tf.concat(current_images, 3)
    old_input = tf.concat(old_images, 3)
    if use_temporal_loss:
        old_to_current = dataloader.old_to_current

if args.validation_file is not None:
    with tf.variable_scope("validation_inputs"):
        v_ref_image = dataloader_validation.ref_images
        v_current_images = dataloader_validation.current_images
        v_old_images = dataloader_validation.old_images
        v_current_input = tf.concat(v_current_images, 3)
        v_old_input = tf.concat(v_old_images, 3)
        if  use_temporal_loss:
            v_old_to_current = dataloader_validation.old_to_current

# Produce outputs from the blending network
with tf.variable_scope("model", reuse=False):
    current_out, current_img_list = \
        blending_network(current_input, current_images, num_input_layers, use_global_mesh, direct_regression, batch_size)
    warped_current = current_out

with tf.variable_scope("model", reuse=True):
    old_out, old_img_list = \
        blending_network(old_input, old_images, num_input_layers, use_global_mesh, direct_regression, batch_size)

if use_temporal_loss:
    warped_current = bilinear_sampler_2d(current_out, old_to_current, 'edge')

if args.validation_file is not None:
    with tf.variable_scope("model", reuse=True):
        v_current_out, v_current_img_list = \
            blending_network(v_current_input, v_current_images, num_input_layers, use_global_mesh, direct_regression, batch_size)
        v_warped_current = v_current_out

    with tf.variable_scope("model", reuse=True):
        v_old_out, v_old_img_list = \
            blending_network(v_old_input, v_old_images, num_input_layers, use_global_mesh, direct_regression, batch_size)

    if use_temporal_loss:
        v_warped_current = bilinear_sampler_2d(v_current_out, v_old_to_current, 'edge')

# Define losses for training
standard_loss = compute_loss(current_out, ref_image, 2)
temporal_loss = compute_loss(warped_current, old_out, 2)
minimize_loss = standard_loss
if use_temporal_loss:
    minimize_loss += temporal_alpha * temporal_loss

if args.validation_file is not None:
    v_standard_loss = compute_loss(v_current_out, v_ref_image, 2)
    v_temporal_loss = compute_loss(v_warped_current, v_old_out, 2)

# Train the network using Adam
global_step = tf.get_variable("global_step", initializer=tf.constant(0, dtype=tf.int32), trainable=False)

ibr_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model")
update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(3e-4).minimize(minimize_loss, global_step=global_step, var_list=ibr_train_vars)

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

# Surround an image with a completely white (1, 1, 1)
# and black (0, 0, 0) line, to make sure that tensorboard
# doesn't normalize each image idependently.
def tensorboard_normalize_hack(img):
    return tf.concat([tf.zeros_like(img[:,0:1,:,:]),
                      img,
                      tf.ones_like(img[:,0:1,:,:])], axis=1)

# Visualize results in tensorboard
vis_output    = tf.summary.image("result",    tensorboard_normalize_hack(current_out), max_outputs=batch_size)
vis_reference = tf.summary.image("reference", tensorboard_normalize_hack(ref_image),   max_outputs=batch_size)
vis_previous  = tf.summary.image("previous",  tensorboard_normalize_hack(old_out),     max_outputs=batch_size)

if use_temporal_loss:
    vis_warped_current = tf.summary.image("current_warped", tensorboard_normalize_hack(warped_current), max_outputs=batch_size)
    if args.validation_file is not None:
        vis_v_warped_current = tf.summary.image("validation_current_warped", tensorboard_normalize_hack(v_warped_current), max_outputs=batch_size)

if args.validation_file is not None:
    vis_v_output    = tf.summary.image("validation_result",    tensorboard_normalize_hack(v_current_out), max_outputs=batch_size)
    vis_v_reference = tf.summary.image("validation_reference", tensorboard_normalize_hack(v_ref_image),   max_outputs=batch_size)
    vis_v_previous  = tf.summary.image("validation_previous",  tensorboard_normalize_hack(v_old_out),     max_outputs=batch_size)

if debug_mode:
    image_index = 0
    if use_global_mesh:
        vis_g_c = tf.summary.image("input_textured_mesh",
                                   tensorboard_normalize_hack(current_img_list[image_index]),
                                   max_outputs=batch_size)
        image_index = image_index + 1
    if num_input_layers > 0:
        vis_a_c = tf.summary.image("input_00_mosaic",
                                   tensorboard_normalize_hack(current_img_list[image_index]),
                                   max_outputs=batch_size)
        image_index = image_index + 1
    if num_input_layers > 1:
        vis_b_c = tf.summary.image("input_01_mosaic",
                                   tensorboard_normalize_hack(current_img_list[image_index]),
                                   max_outputs=batch_size)
        image_index = image_index + 1
    if num_input_layers > 2:
        vis_c_c = tf.summary.image("input_02_mosaic",
                                   tensorboard_normalize_hack(current_img_list[image_index]),
                                   max_outputs=batch_size)
        image_index = image_index + 1
    if num_input_layers > 3:
        vis_d_c = tf.summary.image("input_03_mosaic",
                                   tensorboard_normalize_hack(current_img_list[image_index]),
                                   max_outputs=batch_size)

    vis_loss = tf.summary.image("image_error", compute_error(current_out, ref_image, 2), max_outputs=batch_size)

train_summary = tf.summary.merge([tf.summary.scalar("batch_loss", standard_loss),
                                    tf.summary.scalar("temporal_loss", temporal_loss)])

if args.validation_file is not None:
    validation_summary = tf.summary.merge([tf.summary.scalar("validation_batch_loss", v_standard_loss),
                                            tf.summary.scalar("validation_temporal_loss", v_temporal_loss)])

log_path   = os.path.join(args.output_path, args.log_path)
model_path = os.path.join(args.output_path, args.model_path)
if not os.path.isdir(log_path):
    os.makedirs(log_path, 0o755)
if not os.path.isdir(model_path):
    os.makedirs(model_path, 0o755)
summary_writer = tf.summary.FileWriter(log_path, sess.graph)

# Load pretrained weights
first_batch = 0
last_checkpoint = tf.train.latest_checkpoint(model_path)
if last_checkpoint != None:
    train_variables = []
    for var in tf.global_variables():
        train_variables.append(var)
    train_restorer = tf.train.Saver(train_variables)
    train_restorer.restore(sess, last_checkpoint)
    first_batch = int(last_checkpoint.split("-")[-1])
    print("Restarting from batch: " + str(first_batch))
elif image_loss == 'VGG' or image_loss == "VGG_AND_L1": # New fresh start, we only need to load the VGG weights
    vgg_variables = []
    for var in tf.global_variables():
        if var.name.startswith("vgg_16") and var.name.find("Adam") == -1:
            vgg_variables.append(var)
    vgg_restorer = tf.train.Saver(vgg_variables)
    vgg_restorer.restore(sess, "vgg_16.ckpt")

# Start the data loader threads
coordinator = tf.train.Coordinator()
threads     = tf.train.start_queue_runners(sess=sess, coord=coordinator)

# Main training loop
with sess.as_default():
    try:
        loss_dump_interval  = 8
        image_dump_interval = 256
        for i in range(first_batch, num_batches):
            if coordinator.should_stop():
                break

            run_list = [train_step]
            if i == 0 or i % loss_dump_interval == (loss_dump_interval - 1):
                run_list = run_list + [standard_loss, train_summary]
                if args.validation_file is not None:
                    run_list += [validation_summary]

            if i == 0 or i % image_dump_interval == (image_dump_interval - 1):
                run_list = run_list + [vis_output, vis_reference, vis_previous]
                if args.validation_file is not None:
                    run_list += [vis_v_output, vis_v_reference, vis_v_previous]

                if use_temporal_loss:
                    run_list = run_list + [vis_warped_current]
                    if args.validation_file is not None:
                        run_list += [vis_v_warped_current]

                if debug_mode:
                    run_list = run_list + [vis_loss]
                    if use_global_mesh:
                        run_list = run_list + [vis_g_c]
                    if num_input_layers > 0:
                        run_list = run_list + [vis_a_c]
                    if num_input_layers > 1:
                        run_list = run_list + [vis_b_c]
                    if num_input_layers > 2:
                        run_list = run_list + [vis_c_c]
                    if num_input_layers > 3:
                        run_list = run_list + [vis_d_c]

            output = sess.run(run_list)

            if i == 0 or i % loss_dump_interval == (loss_dump_interval - 1):
                print("Step: %d, batch loss: %f" % (i , output[1]))

            for j in range(2, len(output)):
                summary_writer.add_summary(output[j], global_step=i)
        
            if i % image_dump_interval == (image_dump_interval - 1):
                print("saving model")
                saver.save(sess, os.path.join(model_path, "model"), global_step=global_step)
    except Exception as e:
        # Report exceptions to the coordinator.
        coordinator.request_stop(e)
    finally:
        coordinator.request_stop()
        coordinator.join(threads)
