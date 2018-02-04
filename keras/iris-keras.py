"""
Version 20170125 by Jian: follow the tutorial
http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


Version 20170127 by Jian: test on server
ImportError: cannot import name model_selection
due to
>>> sklearn.__version__
'0.15.2'

Version 20170128 by Jian: test on home laptop
Version 20170212 by Jian: use keras for iris
Version 20170220 by Jian: recap, revisit keras, *packaging
Version 20170304 by Jian: review CV
Version 20170317 by Jian: test multiple gpu => Not sure whether it works
Version 20170402 by Jian: rerun without turning on >1 gpu
Version 20170404 by Jian: rerun turning on >1 gpu
Version 20170430 by Jian: revamp
Version 20170719 by Jian: revamp multi gpu
"""





#
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#csv_path='data/iris.csv'
csv_path='../data/iris.csv'


import pandas

def acquire_data():
	df= pandas.read_csv(url,header=None)
	df.to_csv(csv_path,header=False,index=False)

#acquire_data()



df= pandas.read_csv(csv_path,header=None)

print(df.head())



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

from keras.models import Sequential
from keras.layers import Dense

# baseline test : multinomial regression
model = Sequential()
#model.add(Dense(3,input_dim=4,activation='sigmoid'))
model.add(Dense(3,input_dim=4,activation='softmax'))

#################################################################################################################
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    #return [x.name for x in local_device_protos if x.device_type == 'GPU']
    return [(x.name,x.device_type) for x in local_device_protos]

print(get_available_gpus())



# ref https://github.com/fchollet/keras/issues/2436
# Minimal multi-gpu training by data paralellism
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers.core import Lambda

def slice_batch(x, n_gpus, part):
	sh = K.shape(x)
	L = sh[0] // n_gpus
	if part == n_gpus - 1:
		return x[part*L:]
	return x[part*L:(part+1)*L]
def to_multi_gpu(model, n_gpus=2):
	with tf.device('/cpu:0'):
		x = Input(model.input_shape[1:])#, name=model.input_names[0])
	towers = []
	for g in range(n_gpus):
		print('/gpu:' + str(g))
		with tf.device('/gpu:' + str(g)):
			slice_g = Lambda(slice_batch,  arguments={'n_gpus':n_gpus, 'part':g})(x) 
			towers.append(model(slice_g))
	with tf.device('/cpu:0'):
		merged = concatenate(towers, axis=0)
	return Model(inputs=[x], outputs=merged)

#################################################################################################################
#K.gpu_setup = ["gup0", "gpu1"]

model = to_multi_gpu(model,n_gpus=2)


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X,dummy_y,epochs=10)
#pred = model.predict(X)
#print(pred)


#pred_prob = model.predict_prob(X)
#pred_cls = model.predict_classes(X)
#import numpy as np
#print(np.concatenate((encoded_Y.reshape(150,1),pred_cls.reshape(150,1)),axis=1))
#print(np.sum(pred_proba,axis=1))
quit()




# LOCAL only: on server
import os
import sys
# force to use local packages, in particular, sklearn
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')



















from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
from sklearn.model_selection import KFold
seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(results)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
run on local:
Using TensorFlow backend.
[ 1.          0.93333334  0.93333334  1.          0.93333334  1.          1.
  0.93333334  0.93333334  0.86666667]
Baseline: 95.33% (4.27%)

run on server:
Using Theano backend.
[ 0.53333334  0.93333334  1.          1.          0.26666667  1.          1.
  0.93333334  0.93333334  0.86666667]
Baseline: 84.67% (23.49%)

[ 1.          0.93333334  1.          1.          1.          0.93333334
  1.          0.93333334  1.          0.86666667]
Baseline: 96.67% (4.47%)

[ 1.          0.93333334  1.          1.          1.          1.          1.
  0.93333334  1.          0.86666667]
Baseline: 97.33% (4.42%)
'''
