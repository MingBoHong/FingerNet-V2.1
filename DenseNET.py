import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import config
import numpy as np


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network


def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size,
                                       strides=stride)  # The stride value does not matter


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)  # concate channel


def Linear(x):
    return tf.layers.dense(inputs=x, units=config.class_num, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):

        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=config.dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=config.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=config.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[5, 5], stride=1, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[2, 2], stride=2)

        for i in range(self.nb_blocks):
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_' + str(i))
            x = self.transition_layer(x, scope='trans_' + str(i))

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)
        return x


def loss(logits, labels):
    labels = tf.cast(x=labels, dtype=tf.int32)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_loss')
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables("linear")])
    tf.add_to_collection(name='losses', value=cross_entropy_loss + l2_loss * config.weight_decay)
    return tf.add_n(inputs=tf.get_collection(key='losses'), name='total_loss')


def one_step_train(total_loss, step):
    # lr = tf.train.exponential_decay(learning_rate=config.initial_learning_rate,
    #                                 global_step=step,
    #                                 decay_steps=config.decay_step,
    #                                 decay_rate=config.learning_rate_decay_factor,
    #                                 staircase=True)
    # tf.summary.scalar("learning_rate", lr)
    #trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    trainer = tf.train.AdamOptimizer(learning_rate=config.init_learning_rate, epsilon=config.epsilon)
    gradient_pairs = trainer.compute_gradients(loss=total_loss)
    gradient_update = trainer.apply_gradients(grads_and_vars=gradient_pairs, global_step=step)  # 返回一步梯度更新操作
    return gradient_update


def get_batch_samples(img_obj, batch_size, shuffle_flag):
    if shuffle_flag == False:
        image_batch, label_batch = tf.train.batch(tensors=img_obj,
                                                  batch_size=batch_size,
                                                  num_threads=1,
                                                  capacity=512)
    else:
        image_batch, label_batch = tf.train.shuffle_batch(tensors=img_obj,
                                                          batch_size=batch_size,
                                                          num_threads=1,
                                                          min_after_dequeue=128,
                                                          capacity=512)

    tf.summary.image("images", image_batch)
    return image_batch, tf.reshape(label_batch, shape=[batch_size])


def read_and_decode(filename, batch_size, flag):
    # 根据文件名生成一个队列
    if flag:
        filename_queue = tf.train.string_input_producer([filename], num_epochs=config.epochs)
    else:
        filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  #
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [config.pic_size, config.pic_size, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return get_batch_samples([img, label], batch_size, shuffle_flag=flag)


def Accuracy(predict_group):
    total = np.sum(predict_group)
    Acc = total/config.batch_size
    return Acc
