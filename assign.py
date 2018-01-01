import tensorflow as tf
State = tf.Variable(0, name="counter")  # 初期値と名前を指定し、変数を定義
one = tf.constant(1)                    # 定数定義
new_value = tf.add(State, one)
update = tf.assign(State, new_value)    # 変数に値を代入 State = State + 1
# init_op = tf.initialize_all_variables() #古い書き方
init_op = tf.global_variables_initializer() # 変数の初期化
sess = tf.Session()
sess.run(init_op)
print(sess.run(State))
sess.run(update)
print(sess.run(State))
