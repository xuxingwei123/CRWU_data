
import tensorflow as tf
import config
import numpy as np
from config import identity_block
FLAGS = config.FLAGS



class LSTMOCR(object):

    def __init__(self):

        self.input_1 = tf.placeholder(tf.float32, [None, FLAGS.image_height,FLAGS.image_width,FLAGS.image_channel], name='input_1')   #输入   [batch_size, heightm width]  batch*6*4000
        self.input_2 = tf.placeholder(tf.float32,[None, FLAGS.image_height,FLAGS.image_width, FLAGS.image_channel], name='input_2')
        self.labels = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [128, 128, 256, 256, FLAGS.out_channels]
        strides = [2, 1]
        count_ = 0
        min_size = min(FLAGS.image_height, FLAGS.image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count_ += 1
        assert (FLAGS.cnn_count <= count_, "FLAGS.cnn_count should be <= {}!".format(count_))

        #=====================parallel CNN part=======================
        with tf.variable_scope('parallel'):
            print(self.input_1)
            #=============对第一个输入进行卷积
            conv_input1_x1 = tf.layers.conv2d(
                inputs=self.input_1,
                filters=32,
                kernel_size=[1, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='conv_input1_x1')
            pool_input1_x1 = tf.layers.max_pooling2d(inputs=conv_input1_x1, pool_size=[1, 2], strides=[1,2], name='pool_input1_x1')
            conv_input1_x2 = tf.layers.conv2d(
                inputs=pool_input1_x1,
                filters=64,
                kernel_size=[1, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='conv_input1_x2')
            pool_input1_x2 = tf.layers.max_pooling2d(inputs=conv_input1_x2, pool_size=[1, 2], strides=[1,2], name='pool_input1_x2')


            conv_input2_x1 = tf.layers.conv2d(
                inputs=self.input_2,
                filters=32,
                kernel_size=[1, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='conv_input2_x1')
            pool_input2_x1 = tf.layers.max_pooling2d(inputs=conv_input2_x1, pool_size=[1, 2], strides=[1,2], name='pool_input2_x1')
            conv_input2_x2 = tf.layers.conv2d(
                inputs=pool_input2_x1,
                filters=64,
                kernel_size=[1, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), name='conv_input2_x2')
            pool_input2_x2 = tf.layers.max_pooling2d(inputs=conv_input2_x2, pool_size=[1, 2], strides=[1,2], name='pool_input2_x2')

            parallel_cnn_out = tf.concat((pool_input1_x2,pool_input2_x2), axis=3)
            print('paller_out shape: {}'.format(parallel_cnn_out.get_shape().as_list()))




        # =====================CNN part=========================
        with tf.variable_scope('cnn'):

            x = parallel_cnn_out
            for i in range(FLAGS.cnn_count):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = self._conv2d(x, 'cnn-%d' % (i + 1), [1,3], filters[i], filters[i + 1], strides[0])
                    x = self._batch_norm('bn%d' % (i + 1), x)
                    x = self._leaky_relu(x, FLAGS.leakiness)
                    x = self._max_pool(x, 2, strides[1])
            self.cnn_out = x
            print('CNN out shape: {}'.format(self.cnn_out.get_shape().as_list()))

        resnet = identity_block(self.cnn_out, kernel_size=3, in_filter=256, out_filters=[4, 250, 256], stage=2, block='b')
        print('resnet_out shape: {}'.format(resnet.get_shape().as_list()))


        with tf.variable_scope('lstm'):


            batch = tf.shape(resnet)

            self.lstm_input = tf.reshape(resnet,[batch[0],batch[2],1*256])
            print("lstm input shape :", format(self.lstm_input.get_shape().as_list()))


            cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)

            cell1 = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            #if self.mode == 'train':
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)    #用了两层cell
            initial_state = stack.zero_state(FLAGS.batch_size, dtype=tf.float32)


            outputs,_=tf.nn.bidirectional_dynamic_rnn(cell,cell1,inputs=self.lstm_input,dtype=tf.float32)

            outputs=tf.concat(outputs,2)
            print("lstm output shape :", format(outputs.get_shape().as_list()))


        with tf.variable_scope('dense'):
            outputs = tf.reshape(outputs, [-1, 13*FLAGS.num_hidden*2])  # [batch_size * max_stepsize, FLAGS.num_hidden]
            self.dense_input = outputs
            print("output shape :", format(outputs.get_shape().as_list()))

            W = tf.get_variable(name='W_out',
                                shape=[13*FLAGS.num_hidden*2, 4],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=4,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b

            print("lstm_out shape :",format(self.logits.get_shape().as_list()))


    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)


        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate).minimize(self.loss,global_step=self.global_step)
        self.pre = tf.cast(tf.argmax(self.logits, 1), tf.int32)

        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.labels)
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.confusion_matrix = tf.confusion_matrix(self.labels,tf.cast(tf.argmax(self.logits, 1), tf.int32), num_classes = 4)

        tf.summary.scalar('loss', self.loss)



    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size[0], filter_size[1], in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = \
                tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=0.9,
                    center=True,
                    scale=True,
                    epsilon=1e-5,
                    updates_collections=None,
                    is_training=True,
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=True,
                    scope='BatchNorm'
                )

        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
