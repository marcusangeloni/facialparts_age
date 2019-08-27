#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Mon 4 Feb 2019 19:00:00

from __future__ import division

import tensorflow as tf
import json
import os
import math
import numpy as np
from stream_net import StreamNet

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    name = 'model_dir', default = './models',
    help = 'Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    name = 'resize_config_file', default = None,
    help = 'Config file with sizes for each part of face')
tf.app.flags.DEFINE_string(
    name = 'train_metadata', default = None,
    help = 'Regex that matches the training tf-records')
tf.app.flags.DEFINE_string(
    name = 'eval_metadata', default = None,
    help = 'Regex that matches the evaluation tf-records')
tf.app.flags.DEFINE_integer(
    name = 'total_steps', default = 1, help = 'Steps to train')
tf.app.flags.DEFINE_integer(
    name = 'batch_size', default = 1, help = 'Batch size')
tf.app.flags.DEFINE_float(
    name = 'learning_rate', default = 1e-3, help = 'Learning rate')
tf.app.flags.DEFINE_float(
    name = 'reg_val', default = 1e-4, help = 'Regularization loss')
tf.app.flags.DEFINE_float(
    name = 'dropout', default = 0.4, help = 'Dropout rate')
tf.app.flags.DEFINE_integer(
    name = 'batch_prefetch', default = 1, help = 'Batch prefetch')
tf.app.flags.DEFINE_integer(
    name = 'n_classes', default = 8, help = 'Number of classes')

def model_fn():
    def _model_fn(features, labels, mode, params):
        batch_eyebrows = tf.map_fn(preproc_eyebrows, features['eyebrows'], tf.float32)
        batch_eyes = tf.map_fn(preproc_eyes, features['eyes'], tf.float32)
        batch_nose = tf.map_fn(preproc_nose, features['nose'], tf.float32)
        batch_mouth = tf.map_fn(preproc_mouth, features['mouth'], tf.float32)

        _input = {
            "eyebrows": batch_eyebrows,
            "eyes": batch_eyes,
            "nose": batch_nose,
            "mouth": batch_mouth
        }

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        logits, endpoints = StreamNet(_input, is_training, FLAGS.n_classes)

        # Loss, training and eval operations are not needed during inference.
        total_loss = None
        loss = None
        train_op = None
        eval_metric_ops = {}
        export_outputs = None

        prediction_dict = {
            'class_whole': tf.argmax(logits, axis = 1, name = 'pred_whole'),
            'class_eyebrows':tf.argmax(endpoints['eyebrows_dense1'], axis = 1, name = 'pred_eyebrows'),
            'class_eyes':tf.argmax(endpoints['eyes_dense1'], axis = 1, name = 'pred_eyes'),
            'class_nose':tf.argmax(endpoints['nose_dense1'], axis = 1, name = 'pred_nose'),
            'class_mouth':tf.argmax(endpoints['mouth_dense1'], axis = 1,name = 'pred_mouth')
        }

        prediction_logits = {
            'whole': logits,
            'pred_eyebrows':endpoints['eyebrows_dense4'],
            'pred_eyes':endpoints['eyes_dense4'],
            'pred_nose':endpoints['nose_dense4'],
            'pred_mouth':endpoints['mouth_dense4']
        }

        if mode != tf.estimator.ModeKeys.PREDICT:

            # It is very important to retrieve the regularization losses
            reg_loss = tf.losses.get_regularization_loss()

            # This summary is automatically caught by the Estimator API
            tf.summary.scalar("Regularization_Loss", tensor = reg_loss)

            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['label'], logits = logits)
            
            loss_eyebrows = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['label'], logits = endpoints['eyebrows_dense4'])
            loss_eyes = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['label'], logits = endpoints['eyes_dense4'])
            loss_nose = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['label'], logits = endpoints['nose_dense4'])
            loss_mouth = tf.losses.softmax_cross_entropy(
                onehot_labels=labels['label'], logits = endpoints['mouth_dense4'])

            tf.summary.scalar("XEntropy_LOSS", tensor = loss)

            total_loss = loss + reg_loss + loss_eyebrows + loss_eyes + loss_nose + loss_mouth

            learning_rate = tf.constant(
                FLAGS.learning_rate, name = 'fixed_learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            vars_to_train = tf.trainable_variables()
            tf.logging.info("Variables to train: {}".format(vars_to_train))

            if is_training:
                # You DO must get this collection in order to perform updates on batch_norm variables
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(
                        loss = total_loss, global_step = tf.train.get_global_step(), var_list = vars_to_train)

            eval_metric_ops = metrics(prediction_logits, labels['label'])

        else:
            export_outputs = {                
                'endpoints':tf.estimator.export.PredictOutput(outputs = {
                    'softmax':tf.nn.softmax(logits),
                    'eyebrows_flatten': endpoints['eyebrows_flatten'],
                    'eyebrows_dense1': endpoints['eyebrows_dense1'],
                    'eyes_flatten': endpoints['eyes_flatten'],
                    'eyes_dense1': endpoints['eyes_dense1'],
                    'nose_flatten': endpoints['nose_flatten'],
                    'nose_dense1': endpoints['nose_dense1'],
                    'mouth_flatten': endpoints['mouth_flatten'],
                    'mouth_dense1': endpoints['mouth_dense1']})                                 
            }

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = prediction_dict,
            loss = total_loss,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops,
            export_outputs = export_outputs)

    return _model_fn

def preproc(image_bytes, width, height):
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.resize_images(image_tensor, (height, width))
    image_tensor = tf.to_float(image_tensor) / 255.0
    image_tensor = tf.reshape(
        image_tensor, [height, width, 3], name = "Reshape_Preproc")

    return image_tensor

def preproc_eyebrows(image_bytes):
    global resize_config
    width, height = resize_config['eyebrows']['width'], resize_config['eyebrows']['height']
    return preproc(image_bytes, width, height)

def preproc_eyes(image_bytes):
    global resize_config
    width, height = resize_config['eyes']['width'], resize_config['eyes']['height']
    return preproc(image_bytes, width, height)

def preproc_nose(image_bytes):
    global resize_config
    width, height = resize_config['nose']['width'], resize_config['nose']['height']
    return preproc(image_bytes, width, height)

def preproc_mouth(image_bytes):
    global resize_config
    width, height = resize_config['mouth']['width'], resize_config['mouth']['height']
    return preproc(image_bytes, width, height)

def input_fn(metadata, batch_size, epochs, batch_prefetch = 5):
    def _parse_function(example_proto):
        """ Parse data from tf.Example. All labels are decoded as float32"""

        parser_dict = {
            "eyebrows": tf.FixedLenFeature((), tf.string, default_value = ""),
            "eyes": tf.FixedLenFeature((), tf.string, default_value = ""),
            "nose": tf.FixedLenFeature((), tf.string, default_value = ""),
            "mouth": tf.FixedLenFeature((), tf.string, default_value = ""),
            "label": tf.FixedLenFeature((), tf.float32, default_value = 0)
        }
        parsed_features = tf.parse_single_example(example_proto, parser_dict)

        return parsed_features

    def _decode_images(features):
        one_hot_label = tf.one_hot(tf.to_int32(
            features['label']), depth = FLAGS.n_classes, dtype = tf.float32)

        return {
            "eyebrows": features["eyebrows"],
            "eyes": features["eyes"],
            "nose": features["nose"],
            "mouth": features["mouth"]
        }, {'label': one_hot_label}

    def _input_fn():
        with tf.name_scope('Data_Loader'):

            dataset = tf.data.TFRecordDataset(
                metadata, compression_type="GZIP")
            dataset = dataset.map(_parse_function)
            dataset = dataset.map(_decode_images)
            dataset = dataset.shuffle(buffer_size = batch_prefetch * batch_size)

            dataset = dataset.repeat(epochs)
            return dataset.batch(batch_size)

    return _input_fn


def metrics(predictions, labels):
    argmax_labels = tf.argmax(labels, axis = 1)
    
    argmax_whole = tf.argmax(predictions['whole'], axis = 1)
    argmax_pred_eyebrows = tf.argmax(predictions['pred_eyebrows'], axis = 1)
    argmax_pred_eyes = tf.argmax(predictions['pred_eyes'], axis = 1)
    argmax_pred_nose = tf.argmax(predictions['pred_nose'], axis = 1)
    argmax_pred_mouth = tf.argmax(predictions['pred_mouth'], axis = 1)

    return {
        'accuracy_whole': tf.metrics.accuracy(argmax_labels, argmax_whole),
        'accuracy_eyebrows': tf.metrics.accuracy(argmax_labels, argmax_pred_eyebrows),
        'accuracy_eyes': tf.metrics.accuracy(argmax_labels, argmax_pred_eyes),
        'accuracy_nose': tf.metrics.accuracy(argmax_labels, argmax_pred_nose),
        'accuracy_mouth': tf.metrics.accuracy(argmax_labels, argmax_pred_mouth)
    }

def get_serving_fn():
    return tf.estimator.export.build_raw_serving_input_receiver_fn({"eyebrows": tf.placeholder(dtype = tf.string, shape = [None]),
                                                                    "eyes": tf.placeholder(dtype = tf.string, shape = [None]),
                                                                    "nose": tf.placeholder(dtype = tf.string, shape = [None]),
                                                                    "mouth": tf.placeholder(dtype = tf.string, shape = [None])})

def list_tfrecord(regex):
    list_op = tf.train.match_filenames_once(regex)
    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_ops)
        files = sess.run(list_op)

    return files

def get_dataset_len(tfrecord_list, top_n = 1):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    return len(tfrecord_list) * np.mean(
        list(sum(1 for _ in tf.python_io.tf_record_iterator(path, options)) for path in tfrecord_list[:top_n]))

resize_config = {}

def train_streams():
    global resize_config
    resize_config = json.load(open(FLAGS.resize_config_file, 'r'))

    train_metadata = list_tfrecord(FLAGS.train_metadata)
    eval_metadata = list_tfrecord(FLAGS.eval_metadata)

    TRAIN_SET_SIZE = get_dataset_len(train_metadata)
    EVAL_SET_SIZE = get_dataset_len(eval_metadata)
    tf.logging.info('EVAL_SET_SIZE: {}'.format(EVAL_SET_SIZE))

    epochs = int(math.ceil(FLAGS.total_steps /
                           (TRAIN_SET_SIZE / FLAGS.batch_size)))

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    task_data = env.get('task') or {'type': 'master', 'index': 0}
    trial = task_data.get('trial')

    if trial is not None:
        output_dir = os.path.join(FLAGS.model_dir, trial)
        tf.logging.info(
            "Hyperparameter Tuning - Trial {}. model_dir = {}".format(trial, output_dir))
    else:
        output_dir = FLAGS.model_dir

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=1000,
        session_config=session_config,
        save_checkpoints_steps=500,
        save_checkpoints_secs=None,
        keep_checkpoint_max=5
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn(),
        config=run_config
    )

    train_input_fn = input_fn(
        batch_size = FLAGS.batch_size, metadata = train_metadata, epochs = epochs, batch_prefetch = FLAGS.batch_prefetch)
    eval_input_fn = input_fn(
        batch_size = FLAGS.batch_size, metadata = eval_metadata, epochs = 1, batch_prefetch = FLAGS.batch_prefetch)

    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn, max_steps=FLAGS.total_steps)

    eval_steps = math.ceil(EVAL_SET_SIZE / FLAGS.batch_size)
    tf.logging.info('eval steps: {}'.format(eval_steps))

    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_input_fn,
        steps = eval_steps,
        start_delay_secs = 300,
        throttle_secs = 120)

    tf.estimator.train_and_evaluate(
        estimator = estimator, train_spec = train_spec, eval_spec = eval_spec)

    estimator.export_savedmodel(
        export_dir_base = output_dir, serving_input_receiver_fn = get_serving_fn())

if __name__ == "__main__":
    train_streams()
