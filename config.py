"""

"""

import os
import numpy as np
import tensorflow as tf
import random
import pandas as pd



tf.app.flags.DEFINE_boolean('restore',True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 1, 'image height')
tf.app.flags.DEFINE_integer('image_width', 400, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('cnn_count', 3, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 150, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 6300, 'the batch_size')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 100, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 3000, 'the lr decay_step for optimizer')
#tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_data_dir', './data/TRAIN.tsv', 'the train data dir')
tf.app.flags.DEFINE_string('val_data_dir', './data/TEST.tsv', 'the val data dir')
tf.app.flags.DEFINE_string('pre_data_dir', './data/TEST.tsv', 'the pre data dir')
tf.app.flags.DEFINE_string('log_dirs', './log', 'the logging dir')

tf.app.flags.DEFINE_string('mode', 'val', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

FLAGS = tf.app.flags.FLAGS





class DataIterator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_file_name = []
        self.label_name = []
        self.file_num = 0
        self.data_xyz = []
        self.data_label =[]
        f = open(self.data_dir)
        df = pd.read_csv(f,  sep='\t')
        self.all = df.iloc[:,:].values





    def read_data(self, shuffer = True):

        self.all = np.array(self.all)
        if(shuffer == True):
            np.random.shuffle(self.all)
        label = self.all[:,0]
        data1 = self.all[:, 1:401]
        data2 = self.all[:, 401:]
        data1 = np.reshape(data1,[-1,1,400,1])
        data2 = np.reshape(data2, [-1, 1, 400, 1])
        return label, data1, data2


def weight_variable(shape):

    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)

    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)




def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):


    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X + b_conv2)

        # third

        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X + b_conv3)
        # final step
        add = tf.add(X, X_shortcut)
        b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add + b_conv_fin)

    return add_result

def convolutional_block(X_input, kernel_size, in_filter,out_filters, stage, block, stride=2):

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            # first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='SAME')
            b_conv1 = bias_variable([f1])
            X = tf.nn.relu(X + b_conv1)

            # second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            b_conv2 = bias_variable([f2])
            X = tf.nn.relu(X + b_conv2)

            # third
            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            b_conv3 = bias_variable([f3])
            X = tf.nn.relu(X + b_conv3)
            # shortcut path
            W_shortcut = weight_variable([1, 1, in_filter, f3])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')


            add = tf.add(x_shortcut, X)

            b_conv_fin = bias_variable([f3])
            add_result = tf.nn.relu(add + b_conv_fin)

        return add_result
