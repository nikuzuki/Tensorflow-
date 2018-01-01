import tensorflow as tf

# 入力となる定数の定義
x = tf.constant(1, name='x')
y = tf.constant(2, name='y')

add_op = tf.add(x, y)

with tf.Session() as sess:
    print(sess.run(add_op))
