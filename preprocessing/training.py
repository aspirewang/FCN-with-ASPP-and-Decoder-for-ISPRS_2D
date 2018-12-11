import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
slim = tf.contrib.slim


def get_labels_from_annotation(annotation_tensor, class_labels):
    valid_entries_class_labels = class_labels[:-1]
    labels_2d = list(map(lambda x: tf.equal(annotation_tensor, x),
                         valid_entries_class_labels))
    labels_2d_stacked = tf.stack(labels_2d, axis=2)
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)
    return labels_2d_stacked_float


def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    batch_labels = tf.map_fn(fn=lambda x: get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)
    return batch_labels


def get_valid_entries_indices_from_annotation_batch(annotation_batch_tensor, class_labels):
    mask_out_class_label = class_labels[-1]
    valid_labels_mask = tf.not_equal(annotation_batch_tensor,
                                     mask_out_class_label)

    valid_labels_indices = tf.where(valid_labels_mask)
    return tf.to_int32(valid_labels_indices)


def get_valid_logits_and_labels(annotation_batch_tensor,
                                logits_batch_tensor,
                                class_labels):

    labels_batch_tensor = get_labels_from_annotation_batch(annotation_batch_tensor=annotation_batch_tensor,
                                                           class_labels=class_labels)

    valid_batch_indices = get_valid_entries_indices_from_annotation_batch(
        annotation_batch_tensor=annotation_batch_tensor,
        class_labels=class_labels)

    valid_labels_batch_tensor = tf.gather_nd(params=labels_batch_tensor, indices=valid_batch_indices)

    valid_logits_batch_tensor = tf.gather_nd(params=logits_batch_tensor, indices=valid_batch_indices)

    return valid_labels_batch_tensor, valid_logits_batch_tensor
