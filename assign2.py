import tensorflow as tf
State = tf.Variable(0, name = "counter")
one = tf.constant(1)
new_value = tf.add(State, one)
update = tf.assign(State, new_value)
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
sess.run(update)
print(sess.run(State))

sess2 = tf.Session()
sess2.run(init_op)
sess2.run(update)
sess2.run(update)
print(sess2.run(State))
