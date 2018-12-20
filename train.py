import DenseNET
import evaluate
import tensorflow as tf
import os
import numpy as np
import config



def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(initial_value=0,trainable=False)
        img_batch, label_batch = DenseNET.read_and_decode(config.Trainrecords,config.batch_size,flag=True)
        is_training = tf.cast(True, tf.bool)
        logits = DenseNET.DenseNet(x=img_batch, nb_blocks=config.nb_block, filters=config.growth_k, training=is_training).model
        total_loss = DenseNET.loss(logits, label_batch)
        predict_group = tf.nn.in_top_k(predictions=logits, targets=label_batch, k=1)
        one_step_gradient_update = DenseNET.one_step_train(total_loss, global_step)

        saver = tf.train.Saver(var_list=tf.global_variables())
        all_summary_obj = tf.summary.merge_all()

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        step = 0
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord,sess=sess)

            if not os.path.exists(config.event_log_path):
                os.makedirs(config.event_log_path)

            Event_writer = tf.summary.FileWriter(logdir=config.event_log_path, graph=sess.graph)
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            try:
                while not coord.should_stop():
                    _, loss_value,predict = sess.run(fetches=[one_step_gradient_update, total_loss,predict_group])
                    assert not np.isnan(loss_value)
                    if step % 100 == 0:
                        print('step %d, the loss_value is %.2f,Acc is  %.2f' % (step, loss_value,DenseNET.Accuracy(predict)))
                        all_summaries = sess.run(all_summary_obj)
                        Event_writer.add_summary(summary=all_summaries, global_step=step)

                    if step % 1000 == 0:
                        if not os.path.exists(config.checkpoint_path):
                            os.makedirs(config.checkpoint_path)
                        variables_save_path = os.path.join(config.checkpoint_path, 'model-parameters.bin')
                        saver.save(sess, variables_save_path,
                                   global_step=step)
                        evaluate.evaluate()
                    step = step +1
            except tf.errors.OutOfRangeError:
                print('Epochs Complete!')
            finally:
                coord.request_stop()
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()