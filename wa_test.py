import tensorflow as tf
num1 = tf.constant(1)
num2 = tf.constant(2)
num3 = tf.constant(3)
num1PlusNum2 = tf.add(num1,num2)
num1PlusNum2PlusNum3 = tf.add(num1PlusNum2,num3)
sess = tf.Session()
result = sess.run(num1PlusNum2PlusNum3)
print(result)
