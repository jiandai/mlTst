# Ver 20170417 by Jian: hack mnist tutorial data pipe for iris, exercise on data feeding
# Assume there are iris_test.csv, iris_training.csv

import pandas as pd
training_df = pd.read_csv('iris_training.csv',skiprows=1,header=None)
training_features = training_df.iloc[:,0:4].values
training_labels = training_df.iloc[:,4].values

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot
training_labels_1h = dense_to_one_hot(training_labels,3)
training_dataset = DataSet(training_features, training_labels_1h,reshape=False)

import tensorflow as tf
x = tf.placeholder(tf.float32,[None,4])
y0 = tf.placeholder(tf.int8,[None])
y = tf.placeholder(tf.int8,[None,3])
sess = tf.Session()

# Verify 1-hot code works
#print sess.run([y0,y],{y0:training_labels,y:training_labels_1h})
for b in range(3):
	btch_x,btch_y = training_dataset.next_batch(7)
	print b, sess.run([x,y],{x:btch_x,y:btch_y})
