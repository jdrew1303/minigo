# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The policy and value networks share a majority of their architecture.
This helps the intermediate layers extract concepts that are relevant to both
move prediction and score estimation.
"""

import functools
import logging
import numpy as np
import random

from absl import flags
import tensorflow as tf

import features as features_lib
import go
import symmetries


flags.DEFINE_integer('train_batch_size', 256,
                     'Batch size to use for train/eval evaluation. For GPU '
                     'this is batch size as expected. If \"use_tpu\" is set,'
                     'final batch size will be = train_batch_size * num_tpu_cores')

flags.DEFINE_integer('conv_width', 256 if go.N == 19 else 32,
                     'The width of each conv layer in the shared trunk.')

flags.DEFINE_integer('policy_conv_width', 2,
                     'The width of the policy conv layer.')

flags.DEFINE_integer('value_conv_width', 1,
                     'The width of the value conv layer.')

flags.DEFINE_integer('fc_width', 256 if go.N == 19 else 64,
                     'The width of the fully connected layer in value head.')

flags.DEFINE_integer('trunk_layers', go.N,
                     'The number of resnet layers in the shared trunk.')

flags.DEFINE_multi_integer('lr_boundaries', [400000, 600000],
                           'The number of steps at which the learning rate will decay')

flags.DEFINE_multi_float('lr_rates', [0.01, 0.001, 0.0001],
                         'The different learning rates')

flags.DEFINE_integer('training_seed', 0,
                     'Random seed to use for training and validation')

flags.register_multi_flags_validator(
    ['lr_boundaries', 'lr_rates'],
    lambda flags: len(flags['lr_boundaries']) == len(flags['lr_rates']) - 1,
    'Number of learning rates must be exactly one greater than the number of boundaries')

flags.DEFINE_float('l2_strength', 1e-4,
                   'The L2 regularization parameter applied to weights.')

flags.DEFINE_float('value_cost_weight', 1.0,
                   'Scalar for value_cost, AGZ paper suggests 1/100 for '
                   'supervised learning')

flags.DEFINE_float('sgd_momentum', 0.9,
                   'Momentum parameter for learning rate.')

flags.DEFINE_string('work_dir', None,
                    'The Estimator working directory. Used to dump: '
                    'checkpoints, tensorboard logs, etc..')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU for training.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name used'
    'when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_integer(
    'num_tpu_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string('gpu_device_list', None,
                    'Comma-separated list of GPU device IDs to use.')

flags.DEFINE_bool('quantize', False,
                  'Whether create a quantized model. When loading a model for '
                  'inference, this must match how the model was trained.')

flags.DEFINE_integer('quant_delay', 700 * 1024,
                     'Number of training steps after which weights and '
                     'activations are quantized.')

flags.DEFINE_integer(
    'iterations_per_loop', 128,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'summary_steps', default=256,
    help='Number of steps between logging summary scalars.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')

flags.DEFINE_bool(
    'use_random_symmetry', True,
    help='If true random symmetries be used when doing inference.')

flags.DEFINE_bool(
    'use_SE', False,
    help='Use Squeeze and Excitation.')

flags.DEFINE_bool(
    'use_SE_bias', False,
    help='Use Squeeze and Excitation with bias.')

flags.DEFINE_integer(
    'SE_ratio', 2,
    help='Squeeze and Excitation ratio.')

flags.DEFINE_bool(
    'use_swish', False,
    help=('Use Swish activation function inplace of ReLu. '
          'https://arxiv.org/pdf/1710.05941.pdf'))

flags.DEFINE_bool(
    'bool_features', False,
    help='Use bool input features instead of float')

flags.DEFINE_string(
    'input_features', 'agz',
    help='Type of input features: "agz" or "mlperf07"')

flags.DEFINE_string(
    'input_layout', 'nhwc',
    help='Layout of input features: "nhwc" or "nchw"')


# TODO(seth): Verify if this is still required.
flags.register_multi_flags_validator(
    ['use_tpu', 'iterations_per_loop', 'summary_steps'],
    lambda flags: (not flags['use_tpu'] or
                   flags['summary_steps'] % flags['iterations_per_loop'] == 0),
    'If use_tpu, summary_steps must be a multiple of iterations_per_loop')

FLAGS = flags.FLAGS


def get_features_planes():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES_PLANES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES_PLANES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def get_features():
    if FLAGS.input_features == 'agz':
        return features_lib.AGZ_FEATURES
    elif FLAGS.input_features == 'mlperf07':
        return features_lib.MLPERF07_FEATURES
    else:
        raise ValueError('unrecognized input features "%s"' %
                         FLAGS.input_features)


def mg_activation(inputs):
    if FLAGS.use_swish:
        return tf.keras.layers.Activation(tf.nn.swish)(inputs)

    return tf.keras.layers.Activation('relu')(inputs)


def mg_batchn(inputs, bn_axis, training=None, center=True, scale=True):
    return tf.keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=.95,
        epsilon=1e-5,
        center=center,
        scale=scale)(inputs, training=training)


def mg_conv2d(inputs, filters, kernel_size, data_format):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False,
        data_format=data_format,
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength))(inputs)


def residual_inner(inputs, filters, bn_axis, data_format, training=None):
    conv_layer1 = mg_batchn(mg_conv2d(inputs, filters, 3, data_format),
                            bn_axis, training=training)
    initial_output = mg_activation(conv_layer1)
    conv_layer2 = mg_batchn(mg_conv2d(initial_output, filters, 3, data_format),
                            bn_axis, training=training)
    return conv_layer2


def mg_res_layer(inputs, filters, bn_axis, data_format, training=None):
    residual = residual_inner(inputs, filters, bn_axis,
                              data_format, training=training)
    output = tf.keras.layers.Add()([inputs, residual])
    output = mg_activation(output)
    return output


def mg_squeeze_excitation_layer(inputs, filters, bn_axis, data_format, training=None):
    ratio = FLAGS.SE_ratio
    assert filters % ratio == 0

    residual = residual_inner(inputs, filters, bn_axis,
                              data_format, training=training)
    pool = tf.keras.layers.GlobalAveragePooling2D(
        data_format=data_format)(residual)
    fc1 = tf.keras.layers.Dense(
        units=filters // ratio,
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength))(pool)
    squeeze = mg_activation(fc1)

    if FLAGS.use_SE_bias:
        fc2 = tf.keras.layers.Dense(
            units=2*filters,
            kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength))(squeeze)
        def split_gamma_bias(x):
            return tf.split(x, 2, axis=-1)
        gamma, bias = tf.keras.layers.Lambda(split_gamma_bias)(fc2)
    else:
        gamma = tf.keras.layers.Dense(
            units=filters,
            kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength))(squeeze)
        bias = 0

    sig = tf.keras.layers.Activation('sigmoid')(gamma)

    # Explicitly signal the broadcast.
    if data_format == 'channels_last':
        scale = tf.keras.layers.Reshape((1, 1, filters))(sig)
        if FLAGS.use_SE_bias:
            bias = tf.keras.layers.Reshape((1, 1, filters))(bias)
    else:
        scale = tf.keras.layers.Reshape((filters, 1, 1))(sig)
        if FLAGS.use_SE_bias:
            bias = tf.keras.layers.Reshape((filters, 1, 1))(bias)

    def multiply_and_add(args):
        scale, residual, bias = args
        return tf.multiply(scale, residual) + bias

    excitation = tf.keras.layers.Lambda(multiply_and_add)([scale, residual, bias])

    output = tf.keras.layers.Add()([inputs, excitation])
    return mg_activation(output)


def get_model():
    """Builds the Keras model."""
    if FLAGS.input_layout == 'nhwc':
        bn_axis = -1
        data_format = 'channels_last'
        input_shape = [go.N, go.N, get_features_planes()]
    else:
        bn_axis = 1
        data_format = 'channels_first'
        input_shape = [get_features_planes(), go.N, go.N]

    inputs = tf.keras.Input(shape=input_shape, name='pos_tensor')
    x = inputs

    if FLAGS.bool_features:
        x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)

    # Initial block
    x = mg_conv2d(x, FLAGS.conv_width, 3, data_format)
    x = mg_batchn(x, bn_axis)
    x = mg_activation(x)

    # Shared trunk
    for _ in range(FLAGS.trunk_layers):
        if FLAGS.use_SE or FLAGS.use_SE_bias:
            x = mg_squeeze_excitation_layer(
                x, FLAGS.conv_width, bn_axis, data_format)
        else:
            x = mg_res_layer(x, FLAGS.conv_width, bn_axis, data_format)

    # Policy head
    policy_conv = mg_conv2d(x, FLAGS.policy_conv_width, 1, data_format)
    policy_conv = mg_batchn(policy_conv, bn_axis, center=False, scale=False)
    policy_conv = mg_activation(policy_conv)
    policy_flat = tf.keras.layers.Flatten()(policy_conv)
    policy_logits = tf.keras.layers.Dense(
        go.N * go.N + 1,
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength),
        name='policy_logits')(policy_flat)
    policy_output = tf.keras.layers.Softmax(name='policy_output')(policy_logits)

    # Value head
    value_conv = mg_conv2d(x, FLAGS.value_conv_width, 1, data_format)
    value_conv = mg_batchn(value_conv, bn_axis, center=False, scale=False)
    value_conv = mg_activation(value_conv)
    value_flat = tf.keras.layers.Flatten()(value_conv)
    value_fc_hidden = tf.keras.layers.Dense(
        FLAGS.fc_width,
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength))(value_flat)
    value_fc_hidden = mg_activation(value_fc_hidden)
    value_output = tf.keras.layers.Dense(
        1,
        activation='tanh',
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2_strength),
        name='value_output')(value_fc_hidden)

    return tf.keras.Model(inputs=inputs, outputs=[policy_output, value_output])


def maybe_set_seed():
    if FLAGS.training_seed != 0:
        random.seed(FLAGS.training_seed)
        tf.random.set_seed(FLAGS.training_seed)
        np.random.seed(FLAGS.training_seed)


def make_model_metadata(metadata):
    for f in ['conv_width', 'fc_width', 'trunk_layers', 'use_SE', 'use_SE_bias',
              'use_swish', 'input_features', 'input_layout']:
        metadata[f] = getattr(FLAGS, f)
    metadata['input_type'] = 'bool' if FLAGS.bool_features else 'float'
    metadata['board_size'] = go.N
    return metadata
