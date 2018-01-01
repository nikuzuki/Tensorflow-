import tensorflow as tf

# 定数の定義
x = tf.constant(1, name='x')
# プレースホルダーという箱を用意する
y = tf.placeholder(tf.int32, name='y')

# x+yの演算を定義
add_op = tf.add(x, y)

with tf.Session() as sess:
    # プレースホルダには, feed_dictという
    # 仕組みを通じて値を外挿出来る
    print(sess.run(add_op, feed_dict={y:1}))
    print(sess.run(add_op, feed_dict={y:3}))
