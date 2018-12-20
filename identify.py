import tensorflow as tf
from scipy import misc
import math
import DenseNET
import numpy as np
import config
C_size = 32
T = 0
F = 0
checkpoint_path =r"F:\KDR\BRL\project\finger\Finger_net_V2.0\model"

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def pad_image(ori_image):

    if math.ceil(ori_image.shape[0] / C_size) > int(ori_image.shape[0] / C_size):
        pad_height = math.ceil(ori_image.shape[0] / C_size)*32 - ori_image.shape[0]
    else:
        pad_height = 0
    if math.ceil(ori_image.shape[1]/C_size) > int(ori_image.shape[1]/C_size):
        pad_width = math.ceil(ori_image.shape[1] / C_size)*32 - ori_image.shape[1]
    else:
        pad_width = 0
    return np.pad(ori_image, ((math.ceil(pad_height/2), int(pad_height/2)), (math.ceil(pad_width/2), int(pad_width/2))), 'constant', constant_values=255)

def cut_image(image):
    data =[]
    var = 0
    number_slice = (image.shape[0]* image.shape[1])/(C_size*C_size)
    mean  = np.mean(image)
    for row in range(int(image.shape[0] / C_size)):
        for col in range(int((image.shape[1]) / C_size)):
            row_start = row * C_size
            row_end = (row+1) * C_size
            col_start = col * C_size
            col_end = (col+1) * C_size
            var = var + np.var(image[ row_start:row_end, col_start:col_end])
    for row in range(int(image.shape[0] /C_size)):
        for col in range(int((image.shape[1]) /C_size)):
            row_start = row * C_size
            row_end = (row + 1) * C_size
            col_start = col * C_size
            col_end = (col + 1) * C_size

            if  np.var(image[row_start:row_end, col_start:col_end]) > (var / number_slice) * 0.5 \
                    and np.mean(image[row_start:row_end, col_start:col_end])< mean*0.9:
                data.append(image[row_start:row_end, col_start:col_end])
    return np.array(data)

def identify(image):
    with tf.Graph().as_default():
        if len(image.shape) == 3:
            image = rgb2gray(image)
            image = np.squeeze(image)
        pic = pad_image(image)
        data = cut_image(pic)
        image = tf.convert_to_tensor(data)
        image = tf.expand_dims(image,-1)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        is_training = tf.cast(False, tf.bool)
        logits = DenseNET.DenseNet(x=image, nb_blocks=config.nb_block, filters=config.growth_k,
                                   training=is_training).model
        saver = tf.train.Saver()
        with tf.Session() as sess:
            checkpoint_proto = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path)
            if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
                saver.restore(sess, checkpoint_proto.model_checkpoint_path)
            else:
                print('checkpoint file not found!')
                return
            predict = sess.run(tf.argmax(logits,1))
            predict = np.array(predict)

            """
             Vote Mechanism
            """

            Num_0 = sum(predict == 0)
            Num_1 = sum(predict == 1)
            if Num_0 <= Num_1:
                return "Alive"
            else:
                return "Spoof"

