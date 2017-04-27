# Ver 20170417 by Jian: hack mnist tutorial data pipe for iris, exercise on data feeding
# Ver 20170418.1 by Jian: add softmax regression
# Assume there are iris_test.csv, iris_training.csv
# Ver 20170418.2 by Jian: add 1-hidden layer

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

## I/O layers
x = tf.placeholder(tf.float32,[None,4])
y0 = tf.placeholder(tf.int8,[None])
y = tf.placeholder(tf.float32,[None,3])

## I-H1
num1_hidden_nodes = 3
#num1_hidden_nodes = 500
W1 = tf.Variable(tf.zeros([4,num1_hidden_nodes],tf.float32))
b1 = tf.Variable(tf.zeros([1,num1_hidden_nodes],tf.float32))
logits = tf.add(tf.matmul(x,W1),b1)
'''
h1 = tf.nn.relu(logits)

## H1-H2
num2_hidden_nodes = 3
W2 = tf.Variable(tf.zeros([num1_hidden_nodes,num2_hidden_nodes],tf.float32))
b2 = tf.Variable(tf.zeros([1,num2_hidden_nodes],tf.float32))
logits = tf.add(tf.matmul(h1,W2),b2)
h2 = tf.nn.relu(logits)



## H2-O
W3 = tf.Variable(tf.zeros([num2_hidden_nodes,3],tf.float32))
b3 = tf.Variable(tf.zeros([1,3],tf.float32))
logits = tf.add(tf.matmul(h2,W3),b3)
'''
##
p = tf.nn.softmax(logits)
#cross_entropy = tf.reduce_sum(-y*tf.log(p),axis=1)
cross_entropy = tf.reduce_sum(-y*tf.log(p),reduction_indices=[1]) 
loss = tf.reduce_mean(cross_entropy) /tf.log(3.)
loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))/tf.log(3.)

##
solver = tf.train.GradientDescentOptimizer(.8)
train_op = solver.minimize(loss)

##
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,1),tf.argmax(y,1)),tf.float16))


sess = tf.Session()
# Verify 1-hot code works
#print(sess.run([y0,y],{y0:training_labels,y:training_labels_1h}))
#print(sess.run([y0,y],{y0:test_labels,y:test_labels_1h}))

#sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())
for b in range(2000000):
    btch_x,btch_y = training_dataset.next_batch(119)
    sess.run(train_op,{x:btch_x,y:btch_y}) 
    if b % 10000 ==0:
     tr_loss,tr_acc = sess.run([loss,accuracy],{x:btch_x,y:btch_y})
     btch_x,btch_y = test_dataset.next_batch(29)
     tt_loss,tt_acc = sess.run([loss,accuracy],{x:btch_x,y:btch_y})
     print(b,tr_loss,tt_loss,tr_acc,tt_acc)

#btch_x,btch_y = training_dataset.next_batch(59)
#print(sess.run([accuracy],{x:btch_x,y:btch_y}))
#btch_x,btch_y = test_dataset.next_batch(29)
#print(sess.run([accuracy],{x:btch_x,y:btch_y}))
