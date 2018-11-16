from text_cnn import TextCNN
import tensorflow as tf
import time

tf.flags.DEFINE_integer("distance_dim", 5, "Dimension of position vector")
tf.flags.DEFINE_integer("embedding_size", 50, "Dimension of word embedding")
tf.flags.DEFINE_integer("n1", 200, "Hidden layer1")
tf.flags.DEFINE_integer("n2", 100, "Hidden layer2")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 0.0001, "Learning rate")
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")
tf.flags.DEFINE_string("filter_sizes", "3,4,5",
                       "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer(
    "num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer(
    "num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 100,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("evaluate_every", 100,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("window_size", 3, "n-gram")
tf.flags.DEFINE_integer("sequence_length", 204, "max tokens b/w entities")
tf.flags.DEFINE_integer("K", 4, "K-fold cross validation")
tf.flags.DEFINE_float("early_threshold", 0.5, "Threshold to stop the training")
FLAGS = tf.flags.FLAGS


start_time = time.time()
session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                              log_device_placement=FLAGS.log_device_placement)
sess = tf.Session(config=session_conf)

cnn = TextCNN(filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              vec_shape=(FLAGS.sequence_length, FLAGS.embedding_size *
                         FLAGS.window_size + 2 * FLAGS.distance_dim),
              l2_reg_lambda=FLAGS.l2_reg_lambda)
# 定义训练过程
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

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

import os

# 配置models和summaries的存储目录
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "data", timestamp))
print("Writing to {}\n".format(out_dir))

# 把loss和accuracy记录下来
loss_summary = tf.summary.scalar("loss", cnn.loss)
acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
