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
Version 20170414 by Jian: fork to iris-tf.py for exercise tf
"""

import tensorflow as tf


X = tf.placeholder(tf.float32,[None,4])
Y = tf.placeholder(tf.float32,[None,3])
W = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))
logits = tf.matmul(X,W)+b
        
quit()


url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
csv_path='../data/iris.csv'
import pandas

def acquire_data():
	df= pandas.read_csv(url,header=None)
	df.to_csv(csv_path,header=False,index=False)

#acquire_data()
df= pandas.read_csv(csv_path,header=None)

dataset = df.values # return a <class 'numpy.ndarray'> type
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]







from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_y)

quit()





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

