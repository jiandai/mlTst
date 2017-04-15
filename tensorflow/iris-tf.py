# -*- coding: utf-8 -*-
"""
Version 20170125 by Jian: follow the tutorial
http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


Version 20170127 by Jian: test on rescomp 
ImportError: cannot import name model_selection
result:
>>> sklearn.__version__
'0.15.2'

Version 20170128 by Jian: test on home laptop
Version 20170212 by Jian: use keras for iris
Version 20170220 by Jian: recap, revisit keras, *packaging
Version 20170414.1 by Jian: fork to iris-tf.py for exercise tf
Version 20170414.2 by Jian: multinomial regression exercise /w iris dataset

to-do:
. multiple gpu from start
. equivalent W's

. more reading => use batch

"""

# Prepare 1-hot dataset 
csv_path='../data/iris.csv'
import pandas
df= pandas.read_csv(csv_path,header=None)

X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(Y)
dummy_y = encoder.transform(Y)
#dummy_y = pandas.get_dummies(df.iloc[:,4]).values


import tensorflow as tf
# inference
x = tf.placeholder(tf.float32,[None,4])
W = tf.Variable(tf.zeros([4,3]),tf.float32)
b = tf.Variable(tf.zeros([3]),tf.float32)
logits = tf.matmul(x,W)+b
softmax = tf.nn.softmax(logits)
y = tf.placeholder(tf.float32,[None,3])
# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)
# train
train = tf.train.GradientDescentOptimizer(.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(800):
	sess.run(train,feed_dict= {x:X,y:dummy_y})
print(sess.run(loss,feed_dict= {x:X,y:dummy_y}))
pred_prob = sess.run(softmax,feed_dict= {x:X,y:dummy_y})
for j in range(pred_prob.shape[0]):
	print(j,pred_prob[j])
print(sess.run([W]))
print(sess.run([b]))


quit()
#df_1hot = pandas.concat([df.iloc[:,0:4],pandas.get_dummies(df.iloc[:,4])],axis=1)
#df_1hot.to_csv('../data/iris_1hot.csv',header=False)
#print(df_1hot)

#csv_path='../data/iris_1hot.csv'
#import collections
#Dataset = collections.namedtuple('Dataset', ['data', 'target'])
# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
#iris = tf.contrib.learn.datasets.base.load_csv_without_header(csv_path,features_dtype=np.float,target_dtype=np.int,target_column=range(4,7))
# target_column can only be a single number
#iris = Dataset(data=df.iloc[:,0:4].values, target=pandas.get_dummies(df.iloc[:,4]).values)
#print(iris.target)
#quit()



import numpy as np





#dummy_y= tf.one_hot(encoded_Y,3,1,0) 
# TypeError: Value passed to parameter 'indices' has DataType string not in list of allowed values: uint8, int32, int64
# TypeError: The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, or numpy ndarrays.
#Y = df.iloc[:,4].values

#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(df.iloc[:,4].values)
#print encoded_Y

# Not sure how to use OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#encoder.fit(encoded_Y.reshape(-1,1))
#print encoder.transform(encoded_Y.reshape(-1,1))

#url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

def acquire_data():
	df= pandas.read_csv(url,header=None)
	df.to_csv(csv_path,header=False,index=False)

#acquire_data()




# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_y)






from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3,input_dim=4,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,dummy_y)
print(model.summary())


def baseline_model():
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
from sklearn.model_selection import KFold
seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Baseline: 96.67% (4.47%)

