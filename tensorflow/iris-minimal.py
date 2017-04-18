# Ver 20170417 by Jian: hack mnist tutorial data pipe for iris, exercise on data feeding
# Ver 20170418 by Jian: add softmax regression
# Assume there are iris_test.csv, iris_training.csv

import pandas as pd
training_df = pd.read_csv('iris_training.csv',skiprows=1,header=None)
training_features = training_df.iloc[:,0:4].values
training_labels = training_df.iloc[:,4].values
test_df = pd.read_csv('iris_test.csv',skiprows=1,header=None)
test_features = test_df.iloc[:,0:4].values
test_labels = test_df.iloc[:,4].values

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet, dense_to_one_hot
training_labels_1h = dense_to_one_hot(training_labels,3)
training_dataset = DataSet(training_features, training_labels_1h,reshape=False)
test_labels_1h = dense_to_one_hot(test_labels,3)
test_dataset = DataSet(test_features, test_labels_1h,reshape=False)

print(training_dataset.num_examples)
print(test_dataset.num_examples)

import tensorflow as tf

##
x = tf.placeholder(tf.float32,[None,4])
y0 = tf.placeholder(tf.int8,[None])
y = tf.placeholder(tf.float32,[None,3])

##
W = tf.Variable(tf.zeros([4,3],tf.float32))
b = tf.Variable(tf.zeros([1,3],tf.float32))
logits = tf.add(tf.matmul(x,W),b)
p = tf.nn.softmax(logits)
#cross_entropy = tf.reduce_sum(-y*tf.log(p),axis=1)
cross_entropy = tf.reduce_sum(-y*tf.log(p),reduction_indices=[1]) 
loss = tf.reduce_mean(cross_entropy)

##
solver = tf.train.GradientDescentOptimizer(.05)
train_op = solver.minimize(loss)

##
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,1),tf.argmax(y,1)),tf.float16))


sess = tf.Session()
# Verify 1-hot code works
#print(sess.run([y0,y],{y0:training_labels,y:training_labels_1h}))
#print(sess.run([y0,y],{y0:test_labels,y:test_labels_1h}))

#sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())
for b in range(9000):
	btch_x,btch_y = training_dataset.next_batch(17)
	sess.run(train_op,{x:btch_x,y:btch_y})

btch_x,btch_y = test_dataset.next_batch(22)
print(sess.run([accuracy],{x:btch_x,y:btch_y}))
