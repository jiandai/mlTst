import numpy as np
num_examples = 5
num_features = 2
data = np.reshape(np.arange(num_examples*num_features), (num_examples, num_features))
print(data)

import tensorflow as tf
tf.reset_default_graph()
data_t = tf.train.slice_input_producer([tf.constant(data)], num_epochs=1, shuffle=False)
data_b = tf.train.batch([data_t],batch_size=2)
data_b_dbg = tf.Print(tf.constant(-1), [data_b], "Dequeueing from data_batch ")

sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables()) # Older version of tf
tf.get_default_graph().finalize()
tf.train.start_queue_runners()
try:
  while True:
    print(sess.run(data_b_dbg))
except tf.errors.OutOfRangeError as e:
  print "No more inputs."
