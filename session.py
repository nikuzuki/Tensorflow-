import tensorflow as tf
inc = tf.constant(1)
cnt = tf.Variable(0, "counter")

add_op = tf.add(cnt, inc)
up_op = tf.assign(cnt, add_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(up_op))
    print(sess.run(up_op))
    print(sess.run(up_op))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(up_op))
    print(sess.run(up_op))
    print(sess.run(up_op))
