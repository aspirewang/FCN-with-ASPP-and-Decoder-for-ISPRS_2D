import tensorflow as tf
import numpy as np
import os
#from tensorflow.contrib import slim
from preprocessing.read_data import tf_record_parser, rescale_image_and_annotation_by_factor, \
    distort_randomly_image_color, scale_image_with_crop_padding, random_flip_image_and_annotation

TRAIN_DATASET_DIR = './dataset/voc_aug_tfrecords'
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

crop_size = 513
batch_size = 12

training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)
training_dataset = training_dataset.map(rescale_image_and_annotation_by_factor)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, crop_size))
training_dataset = training_dataset.map(random_flip_image_and_annotation)
training_dataset = training_dataset.repeat()
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)
validation_dataset = validation_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, crop_size))
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(batch_size)

iterator = training_dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    images, labels, image_shape = sess.run(next_element)
    print(images.shape, labels.shape)



