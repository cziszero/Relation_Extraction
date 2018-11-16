import CNN
from text_cnn import TextCNN
import data_helpers
import os
import numpy as np
import time
import tensorflow as tf
import datetime

with tf.Graph().as_default():
    start_time = time.time()
    session_conf = tf.ConfigProto(allow_soft_placement=CNN.FLAGS.allow_soft_placement,
                                  log_device_placement=CNN.FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(filter_sizes=list(map(int, CNN.FLAGS.filter_sizes.split(","))),
                      num_filters=CNN.FLAGS.num_filters, vec_shape=(
                          CNN.FLAGS.sequence_length, CNN.FLAGS.embedding_size * CNN.FLAGS.window_size + 2 * CNN.FLAGS.distance_dim),
                      l2_reg_lambda=CNN.FLAGS.l2_reg_lambda)
        # 定义训练过程
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # 记录梯度之和稀疏性，便于观察
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 配置models和summaries的存储目录
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "data", timestamp))
        print("Writing to {}\n".format(out_dir))

        # 把loss和accuracy记录下来
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # 训练过程summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint 目录.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_text_train, y_batch):
            """
            进行一批训练
            """
            feed_dict = {
                cnn.input_x: x_text_train,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: CNN.FLAGS.dropout_keep_prob
            }
            ops = [train_op, global_step, train_summary_op,
                   cnn.loss, cnn.accuracy, cnn.scores]
            _, step, summaries, loss, accuracy, scores = sess.run(
                ops, feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            return loss

        def dev_step(x_text_dev, y_batch):
            """
            在测试集上进行评估
            """
            feed_dict = {
                cnn.input_x: x_text_dev,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy, summaries = sess.run(
                [global_step, cnn.loss, cnn.accuracy, dev_summary_op],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            dev_summary_writer.add_summary(summaries, step)
            return loss

        batch_iter = CNN.get_batches()
        X_val, Y_val = CNN.get_validation_data()
        for batch in batch_iter:
            loss = accuracy = 0.0
            X_train, y_train = zip(*batch)
            X_train, Y_train = np.asarray(X_train), np.asarray(
                y_train)  # 把X_train和y_train转换成ndarry
            train_loss = train_step(X_train, Y_train)
            current_step = tf.train.global_step(sess, global_step)  # 记步数
            if current_step % CNN.FLAGS.evaluate_every == 0:
                print("Evaluation:")
                test_loss = dev_step(np.asarray(X_val), np.asarray(Y_val))
                # 如果测试集loss和训练集loss相差大于early_threshold则退出。
                if abs(test_loss - train_loss) > CNN.FLAGS.early_threshold:
                    exit(0)
            print("")
            if current_step % CNN.FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        print("-------------------")
        print("Finished in time %0.3f" % (time.time() - start_time))
