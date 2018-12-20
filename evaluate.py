import tensorflow as tf
import math
import config
import DenseNET
import numpy as np
Test_number = 4000
batch_size = 10

def eval_once(summary_op, summary_writer, saver, predict_true_or_false):
    with tf.Session() as sess:

        checkpoint_proto = tf.train.get_checkpoint_state(checkpoint_dir=config.checkpoint_path)
        if checkpoint_proto and checkpoint_proto.model_checkpoint_path:
            saver.restore(sess, checkpoint_proto.model_checkpoint_path)  # 恢复模型变量到当前session中
        else:
            print('checkpoint file not found!')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for queue_runner in tf.get_collection(key=tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))

            test_batch_num = math.ceil(Test_number / batch_size)
            iter_num = 0
            true_test_num = 0

            total_test_num = test_batch_num * batch_size
            while iter_num < test_batch_num and not coord.should_stop():
                result_judge = sess.run([predict_true_or_false])
                true_test_num += np.sum(result_judge)
                iter_num += 1
            precision = true_test_num / total_test_num
            print("The test precision is %.3f" % precision)
        except:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)


def evaluate():
    with tf.Graph().as_default() as g:

        img_batch, label_batch = DenseNET.read_and_decode(config.Testrecords,batch_size,flag=False)
        is_training = tf.cast(False, tf.bool)
        logits = DenseNET.DenseNet(x=img_batch, nb_blocks=config.nb_block, filters=config.growth_k, training=is_training).model

        predict_true_or_false = tf.nn.in_top_k(predictions=logits, targets=label_batch, k=1)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir='./event-log-test', graph=g)
        eval_once(summary_op, summary_writer, saver, predict_true_or_false)

