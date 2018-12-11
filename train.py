import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import slim
import json
import network
from preprocessing.read_data import tf_record_parser, rescale_image_and_annotation_by_factor, \
    distort_randomly_image_color, scale_image_with_crop_padding, random_flip_image_and_annotation, \
    download_resnet_checkpoint_if_necessary
from preprocessing import training


# 0=impervious surfaces
# 1=building
# 2=low vegetation
# 3=tree
# 4=car
# 5=background



#初始化参数
class params():
    def __init__(self):
        self.resnet_model = "resnet_v2_50"
        self.number_of_classes = 6
        self.crop_size = 512
        self.batch_size = 6
        self.starting_learning_rate = 1e-5
        self.current_best_val_loss = 99999
        self.batch_norm_epsilon = 1e-5
        self.batch_norm_decay = 0.9997
        self.l2_regularizer = 0.0001
        self.multi_grid = [1, 2, 4]
        self.output_stride = 16
        self.gpu_id = 0
        self.accumulated_validation_miou = 0
        self.contiue_training = False
        self.model_id = 9780

args = params()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

#tfrecord文件路径
DATASET_DIR = './dataset/RGB_512_tfrecords'
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
LOG_FOLDER = './tboard_logs'

#从tfrecord文件建立training和validation的dataset
training_filenames = [os.path.join(DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)
training_dataset = training_dataset.map(rescale_image_and_annotation_by_factor)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, args.crop_size))
training_dataset = training_dataset.map(random_flip_image_and_annotation)
training_dataset = training_dataset.repeat()
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)
validation_dataset = validation_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, args.crop_size))
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(args.batch_size)

#resnet的路径，如果没有文件下载resnet预训练模型
resnet_checkpoints_path = "./resnet/checkpoints/"
download_resnet_checkpoint_if_necessary(resnet_checkpoints_path, args.resnet_model)

handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
batch_images_tf, batch_labels_tf, _ = iterator.get_next()

training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

is_training_tf = tf.placeholder(tf.bool, shape=[])

logits_tf, small_logits_tf = tf.cond(is_training_tf, true_fn= lambda: network.deeplab_v3(batch_images_tf, args, is_training=True, reuse=False),
                    false_fn=lambda: network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=True))

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

cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits_batch_tf,
                                                             labels=valid_labels_batch_tf)

small_cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_small_logits_batch_tf,
                                                                   labels=valid_small_labels_batch_tf)

cross_entropy_tf = tf.reduce_mean(cross_entropies)

small_cross_entropy_tf = tf.reduce_mean(small_cross_entropies)

all_cross_entropy_tf = 0.5 * cross_entropy_tf + 0.5 * small_cross_entropy_tf


predictions_tf = tf.argmax(logits_tf, axis=3)

tf.summary.scalar('cross_entropy', cross_entropy_tf)
tf.summary.scalar('small_cross_entropy', small_cross_entropy_tf)
tf.summary.scalar('all_cross_entropy', all_cross_entropy_tf)


with tf.variable_scope("optimizer_vars"):
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.starting_learning_rate)
    train_step = slim.learning.create_train_op(all_cross_entropy_tf, optimizer, global_step=global_step)

if args.contiue_training:
    process_str_id = str(args.model_id)
else:
    process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
LOG_FOLDER = os.path.join(LOG_FOLDER, process_str_id)

if not os.path.exists(LOG_FOLDER):
    print("Tensoboard folder:", LOG_FOLDER)
    os.makedirs(LOG_FOLDER)
else:
    print("Tensorboard folder:", LOG_FOLDER)

variables_to_restore = slim.get_variables_to_restore(exclude=[args.resnet_model + "/logits", "optimizer_vars",
                                                              "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits",
                                                              "DeepLab_v3/Decoder1_layer", "DeepLab_v3/Decoder2_layer"])

miou, update_op = tf.contrib.metrics.streaming_mean_iou(tf.argmax(valid_logits_batch_tf, axis=1),
                                                        tf.argmax(valid_labels_batch_tf, axis=1),
                                                        num_classes=args.number_of_classes)

tf.summary.scalar('miou', miou)

if args.contiue_training:
    restorer = tf.train.Saver()
else:
    restorer = tf.train.Saver(variables_to_restore)

saver = tf.train.Saver()

current_best_val_loss = np.inf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(LOG_FOLDER + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(LOG_FOLDER + '/val')

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    if args.contiue_training:
        restorer.restore(sess, os.path.join(LOG_FOLDER, "train", 'model.ckpt'))
        print("Model", str(args.model_id), "restored.")
    else:
        try:
            restorer.restore(sess, "./resnet/checkpoints/" + args.resnet_model + ".ckpt")
            print("Model checkpoits for " + args.resnet_model + " restored!")
        except FileNotFoundError:
            print("ResNet checkpoints not found. Please download " + args.resnet_model + " model checkpoints from: https://github.com/tensorflow/models/tree/master/research/slim")

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)
    validation_running_loss = []

    train_steps_before_eval = 100
    validation_steps = 20
    while True:
        training_average_loss = 0
        for i in range(train_steps_before_eval):

            _, global_step_np, train_loss, summary_string = sess.run([train_step,
                                                                      global_step, cross_entropy_tf,
                                                                      merged_summary_op],
                                                                                feed_dict = {is_training_tf:True,
                                                                                  handle:training_handle})
            training_average_loss += train_loss

            if i % 10 == 0:
                train_writer.add_summary(summary_string, global_step_np)

        training_average_loss/=train_steps_before_eval

        sess.run(validation_iterator.initializer)

        validation_average_loss = 0
        validation_average_miou = 0
        for i in range(validation_steps):
            val_loss, summary_string, _ = sess.run([cross_entropy_tf, merged_summary_op, update_op],
                                                   feed_dict={handle: validation_handle,
                                                              is_training_tf: False})
            validation_average_loss += val_loss
            validation_average_miou += sess.run(miou)

        validation_average_loss /= validation_steps
        validation_average_miou /= validation_steps

        validation_running_loss.append(validation_average_loss)
        validation_global_loss = np.mean(validation_running_loss)


        if validation_global_loss < current_best_val_loss:

            save_path = saver.save(sess, LOG_FOLDER + "/train" + "/model.ckpt")
            print("Model checkpoints written! Best average val loss:", validation_global_loss)
            current_best_val_loss = validation_global_loss

            args.current_best_val_loss = current_best_val_loss
            with open(LOG_FOLDER + "/train/" + 'data.json', 'w') as fp:
                json.dump(args.__dict__, fp, sort_keys=True, indent=4)

        print("Global step:", global_step_np, "Average train loss:",
              training_average_loss, "\tGlobal Validation Avg Loss:", validation_global_loss,
              "MIoU:", validation_average_miou)

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()