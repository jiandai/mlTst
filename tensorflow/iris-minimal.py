# Ver 20170417 by Jian: hack mnist tutorial data pipe for iris, exercise on data feeding
# Ver 20170418.1 by Jian: add softmax regression
# Assume there are iris_test.csv, iris_training.csv
# Ver 20170418.2 by Jian: add 1-hidden layer
# Ver 20170426.1 by Jian: deep dive to multinomial and 1-hidden layer MLP (tuning weight initialization and learning rate)
# Ver 20170426.2 by Jian: hard to get 2-hidden layer MLP
# Ver 20170427 by Jian: => modularize, => search space, => explore the gradient

import pandas as pd
training_df = pd.read_csv('iris_training.csv',skiprows=1,header=None)
print(training_df.describe())

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

#lr = .8 # for multinomial
#lr = .1 # for 1-hidden MLP, 3 hidden nodes (4-3-3)
lr = .06 # for 2-hidden MLP, 4+4 hidden nodes (4-4-4-3)
STEPS = 2000000 # => to get good prediction for multinomial reg
CHECK=10000
#STEPS = 20
#CHECK =5




## I/O layers
x = tf.placeholder(tf.float32,[None,4])
y0 = tf.placeholder(tf.int8,[None])
y = tf.placeholder(tf.float32,[None,3])





## I-H1
num1_hidden_nodes = 4
#num1_hidden_nodes = 500
W1 = tf.Variable(tf.random_normal([4,num1_hidden_nodes],stddev=.4))
#W1 = tf.Variable(tf.zeros([4,num1_hidden_nodes],tf.float32)) # for ReLU activation, weights cannot be initialized to be zero!
b1 = tf.Variable(tf.zeros([1,num1_hidden_nodes],tf.float32))
logits = tf.add(tf.matmul(x,W1),b1)

h1 = tf.nn.relu(logits)

## H1-H2
num2_hidden_nodes = 4
W2 = tf.Variable(tf.random_normal([num1_hidden_nodes,num2_hidden_nodes],stddev=.4))
#W2 = tf.Variable(tf.zeros([num1_hidden_nodes,num2_hidden_nodes],tf.float32))
b2 = tf.Variable(tf.zeros([num2_hidden_nodes],tf.float32))
logits = tf.add(tf.matmul(h1,W2),b2)

h2 = tf.nn.relu(logits)



## H2-O
W3 = tf.Variable(tf.random_normal([num2_hidden_nodes,3],stddev=.4))
b3 = tf.Variable(tf.zeros([1,3],tf.float32))
logits = tf.add(tf.matmul(h2,W3),b3)

##
p = tf.nn.softmax(logits)
#cross_entropy = tf.reduce_sum(-y*tf.log(p),axis=1)
cross_entropy = tf.reduce_sum(-y*tf.log(p),reduction_indices=[1]) 
loss = tf.reduce_mean(cross_entropy) /tf.log(3.)
#loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))/tf.log(3.)

##
solver = tf.train.GradientDescentOptimizer(lr)
train_op = solver.minimize(loss)

##
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,1),tf.argmax(y,1)),tf.float16))


sess = tf.Session()

#sess.run(tf.global_variables_initializer()) # tf1.x
sess.run(tf.initialize_all_variables())


for b in range(STEPS):
    btch_x,btch_y = training_dataset.next_batch(119)
    sess.run(train_op,{x:btch_x,y:btch_y}) 
    if b % CHECK==0:
     tr_loss,tr_acc = sess.run([loss,accuracy],{x:btch_x,y:btch_y})
     btch_x,btch_y = test_dataset.next_batch(29)
     tt_loss,tt_acc = sess.run([loss,accuracy],{x:btch_x,y:btch_y})
     print(b,tr_loss,tt_loss,tr_acc,tt_acc)




#btch_x,btch_y = training_dataset.next_batch(59)
#print(sess.run([accuracy],{x:btch_x,y:btch_y}))
#btch_x,btch_y = test_dataset.next_batch(29)
#print(sess.run([accuracy],{x:btch_x,y:btch_y}))

# Last a few steps to train the multinomial model
'''
1700000 0.191482 0.206004 0.94971 1.0
1710000 0.18852 0.197269 0.94971 1.0
1720000 0.189373 0.196773 0.94971 1.0
1730000 0.189585 0.210621 0.94971 1.0
1740000 0.189555 0.193945 0.94971 1.0
1750000 0.189422 0.204067 0.94971 1.0
1760000 0.186313 0.204951 0.94971 1.0
1770000 0.187423 0.209739 0.94971 1.0
1780000 0.186846 0.203856 0.94971 1.0
1790000 0.187545 0.204138 0.94971 1.0
1800000 0.187124 0.203793 0.94971 1.0
1810000 0.186678 0.203878 0.94971 1.0
1820000 0.186397 0.205272 0.94971 1.0
1830000 0.182868 0.204192 0.94971 1.0
1840000 0.184038 0.19435 0.94971 1.0
1850000 0.185088 0.198523 0.94971 1.0
1860000 0.184676 0.200762 0.94971 1.0
1870000 0.184289 0.189843 0.94971 1.0
1880000 0.183804 0.192314 0.94971 1.0
1890000 0.179731 0.186551 0.94971 1.0
1900000 0.181728 0.196162 0.94971 1.0
1910000 0.182809 0.201939 0.94971 1.0
1920000 0.181695 0.187372 0.94971 1.0
1930000 0.180564 0.190858 0.94971 1.0
1940000 0.177641 0.190521 0.94971 1.0
1950000 0.181314 0.199237 0.94971 1.0
1960000 0.180657 0.199576 0.94971 1.0
1970000 0.178109 0.192682 0.94971 1.0
1980000 0.180178 0.192308 0.94971 1.0
1990000 0.175074 0.194624 0.94971 1.0
'''
# vs 1-hidden layer MLP
'''
0 0.999811 1.00056 0.35303 0.27588
10000 0.703072 0.758972 0.67236 0.44824
20000 0.298999 0.348955 0.94141 0.93115
30000 0.220377 0.249381 0.94971 0.93115
40000 0.179768 0.192067 0.94971 0.96533
50000 0.203554 0.208544 0.89063 0.93115
60000 0.15158 0.159565 0.94141 0.93115
70000 0.130496 0.123366 0.94971 1.0
80000 0.148524 0.129217 0.94141 0.93115
90000 0.116258 0.105591 0.95801 0.96533
100000 0.118368 0.110186 0.95801 0.96533
110000 0.113263 0.105557 0.95801 0.96533
120000 0.0996693 0.0840765 0.96631 0.96533
130000 0.160172 0.129516 0.90771 0.89648
140000 0.0911758 0.0735151 0.96631 0.96533
150000 0.198713 0.160402 0.90771 0.89648
160000 0.0853511 0.0775344 0.96631 0.96533
170000 0.0917696 0.0831222 0.95801 0.96533
180000 0.0956519 0.0691079 0.94141 0.96533
190000 0.0818826 0.0702104 0.96631 0.96533
200000 0.0935188 0.0885033 0.96631 0.96533
210000 0.0830277 0.0751165 0.97461 0.96533
220000 0.118095 0.0886395 0.94141 0.96533
early stopped
'''

# vs 2-hidden layer MLP
'''
0 0.999367 0.999714 0.35303 0.27588
10000 0.697196 0.774293 0.45386 0.34473
20000 0.273712 0.297442 0.94141 0.96533
30000 0.208352 0.234159 0.94971 0.96533
40000 0.167136 0.165494 0.93262 0.96533
50000 0.144682 0.142151 0.95801 0.96533
60000 0.13296 0.111286 0.95801 0.96533
70000 0.121344 0.111303 0.95801 0.96533
80000 0.118415 0.105218 0.94971 1.0
90000 0.711689 0.610822 0.72266 0.79297
100000 0.103752 0.0804915 0.95801 1.0
110000 0.101834 0.0856768 0.95801 1.0
120000 0.119945 0.10984 0.94971 0.96533
...
early stopped
'''
