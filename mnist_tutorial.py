from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/")
print(mnist)

x = tf.placeholder(tf.float32, [None, 784])
print(x) # Tensor("Placeholder:0", shape=(?, 784), dtype=float32)

# 変数初期化
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b

# 正解ラベル用
y_ = tf.placeholder(tf.int64, [None])
print(y_)

# 交差エントロピー
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 勾配降下アルゴリズムを使って学習率0.5でcross_entropyを最小化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# モデル起動
sess = tf.InteractiveSession()

# 初期化操作
tf.global_variables_initializer().run()

# 訓練ステップ1000回
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# モデル評価 yからはsoftmaxの値が最も高い要素、y_からは正解データ、この2つのインデックスが一致するか
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
print(correct_prediction)

# 精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
