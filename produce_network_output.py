import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os
import argparse
import json
from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training
from metrics import *
from sklearn import metrics
from scipy import io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.interactive(False)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Eval params')
envarg.add_argument("--model_id", default=20096, type=int, help="Model id name to be loaded.")
input_args = parser.parse_args()

# best: 16645
model_name = str(input_args.model_id)

# uncomment and set the GPU id if applicable.
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

# 0=impervious surfaces
# 1=building
# 2=low vegetation
# 3=tree
# 4=car
# 5=background

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

LOG_FOLDER = './tboard_logs'
TEST_DATASET_DIR="./dataset/RGBIRDSM_512_tfrecords"
TEST_FILE = 'validation_no_stride.tfrecords'

test_filenames = [os.path.join(TEST_DATASET_DIR,TEST_FILE)]
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
test_dataset = test_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, args.crop_size))
#test_dataset = test_dataset.shuffle(buffer_size=100)
test_dataset = test_dataset.batch(args.batch_size)

iterator = test_dataset.make_one_shot_iterator()
batch_images_tf, batch_labels_tf, batch_shapes_tf = iterator.get_next()

logits_tf, small_logits_tf =  network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=False)

small_logits_size_tf = tf.shape(small_logits_tf)[1:3]
batch_labels_tf_dims = tf.expand_dims(batch_labels_tf, axis=3)
small_batch_labels_tf_dims = tf.image.resize_nearest_neighbor(batch_labels_tf_dims, small_logits_size_tf, name='label_downsample_x8')
small_batch_labels_tf = tf.squeeze(small_batch_labels_tf_dims,axis=3)

valid_labels_batch_tf, valid_logits_batch_tf = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels_tf,
    logits_batch_tensor=logits_tf,
    class_labels=class_labels)

valid_small_labels_batch_tf, valid_small_logits_batch_tf = training.get_valid_logits_and_labels(
    annotation_batch_tensor=small_batch_labels_tf,
    logits_batch_tensor=small_logits_tf,
    class_labels=class_labels)

cross_entropies_tf = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tf,
                                                                labels=valid_labels_batch_tf)

small_cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_small_logits_batch_tf,
                                                                   labels=valid_small_labels_batch_tf)

cross_entropy_tf = tf.reduce_mean(cross_entropies_tf)
small_cross_entropy_tf = tf.reduce_mean(small_cross_entropies)

all_cross_entropy_tf = 0.5 * cross_entropy_tf + 0.5 * small_cross_entropy_tf


tf.summary.scalar('cross_entropy', cross_entropy_tf)
tf.summary.scalar('small_cross_entropy', small_cross_entropy_tf)
tf.summary.scalar('all_cross_entropy', all_cross_entropy_tf)

predictions_tf = tf.argmax(logits_tf, axis=3)
probabilities_tf = tf.nn.softmax(logits_tf)

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

test_folder = os.path.join(log_folder, model_name, "test")
train_folder = os.path.join(log_folder, model_name, "train")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
    print("Model", model_name, "restored.")

    mean_IoU = []
    mean_pixel_acc = []
    mean_freq_weighted_IU = []
    mean_acc = []
    n = 0
    while True:
        try:
            batch_images_np, batch_predictions_np, batch_labels_np, batch_shapes_np, summary_string= sess.run([batch_images_tf, predictions_tf, batch_labels_tf, batch_shapes_tf, merged_summary_op])
            heights, widths = batch_shapes_np

            # loop through the images in the batch and extract the valid areas from the tensors
            for i in range(batch_predictions_np.shape[0]):

                label_image = batch_labels_np[i]
                pred_image = batch_predictions_np[i]
                input_image = batch_images_np[i]

                indices = np.where(label_image != 255)
                label_image = label_image[indices]
                pred_image = pred_image[indices]
                input_image = input_image[indices]

                if label_image.shape[0] == 263169:
                    label_image = np.reshape(label_image, (513,513))
                    pred_image = np.reshape(pred_image, (513,513))
                    input_image = np.reshape(input_image, (513,513,5))
                else:
                    label_image = np.reshape(label_image, (heights[i], widths[i]))
                    pred_image = np.reshape(pred_image, (heights[i], widths[i]))
                    input_image = np.reshape(input_image, (heights[i], widths[i], 5))

                print(pred_image.shape)
                n += 1
                io.savemat("network_output/" + str(n) + ".mat", {'network_output':pred_image})


        except tf.errors.OutOfRangeError:
            break

    print("测试结束")
