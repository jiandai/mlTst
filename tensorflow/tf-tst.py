# Ver <20170511 by Jian
# Ver 20170511 by Jian: tensorboard test

import tensorflow as tf
print(tf.__version__)
print(tf.__path__)

# Creates a graph.
m1 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,8.0], shape=[2, 4], name='Mat_1')
m2 = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,7.0,5.5], shape=[4, 2], name='Mat_2')
m3 = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='Mat_3')
f = tf.constant(5.,name='Coeff')
m4 = tf.multiply(tf.add(tf.matmul(m1, m2),m3), f, name='Mat_4')

# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess = tf.Session()
# Runs the op.
print(sess.run(m4))

# Added on 5/11/2017 to test tensorboard
#writer = tf.summary.FileWriter('tmp', sess.graph)
writer = tf.summary.FileWriter('../local/', sess.graph)
writer.close()
