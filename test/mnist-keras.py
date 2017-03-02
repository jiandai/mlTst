'''
ver 20170212 by jian:
	ref http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
ver 20170214 by jian: run on server /w gpu
ver 20170301 by jian: use kaggle source
batch command:
bsub -q gpu -n 4 "python mnist-keras.py"
bsub -q gpu -n 8 "python mnist-keras.py"
'''
# force to use local packages, in particular, sklearn
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')

# use keras dataset
#from keras.datasets import mnist
#(X_train_ref, Y_train_ref), (X_test_ref, Y_test_ref) = mnist.load_data()

# use kaggle dataset
import pandas as pd
trainDF = pd.read_csv(os.path.expanduser("~")+'/myworkspace/kaggle/c/digit-recognizer/download/train.csv')
testDF = pd.read_csv(os.path.expanduser("~")+'/myworkspace/kaggle/c/digit-recognizer/download/test.csv')

# Pipe line :
# break trainDF into X and Y
Y_train1 = trainDF.loc[:,'label'].values # (42000,)
X_train1 = trainDF.loc[:,'pixel0':'pixel783'].values # 42000 * 784
X_test1 = testDF.loc[:,'pixel0':'pixel783'].values #28000 * 784

tr_n,tr_p = X_train1.shape
tt_n,tt_p = X_test1.shape
size = 28

#size * size == tr_p
#True
#size * size == tt_p
#True

# reshape X to n-channel-2d array
import numpy as np
X_train = X_train1.reshape((tr_n,1,size,size)).astype('float32')
X_test = X_test1.reshape((tt_n,1,size,size)).astype('float32')
# Assume the max is 255
X_train = X_train / 255
X_test = X_test / 255


# recode Y by 1-hot encoding
from keras.utils import np_utils
Y_train = np_utils.to_categorical(Y_train1)

num_classes = Y_train.shape[1]


print(X_train.shape)
print(Y_train.shape)
#print(X_train_ref.shape)
#print(Y_train_ref.shape)


# build, and compile 2D ConvNet
seed = 7
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

'''
https://github.com/fchollet/keras/issues/2681
check your ~/.keras/keras.json
if "image_dim_ordering": is "th" and "backend": "theano", your input_shape must be (channels, height, width)
if "image_dim_ordering": is "tf" and "backend": "tensorflow", your input_shape must be (height, width, channels)
'''
from keras import backend as K 
K.set_image_dim_ordering('th') # as running on server now /w th

def baseline_model():
	# create model
	model = Sequential()
	# MLP model
	#model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	#model.add(Dense(num_classes, init='normal', activation='softmax'))
	# CNN model
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, size, size), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def larger_model():
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#model = baseline_model()
model = larger_model()

# train 
N_EPOCH = 10
#model.fit(X_train, Y_train, nb_epoch=N_EPOCH, batch_size=200, verbose=2)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=N_EPOCH, batch_size=200, verbose=2)
#scores = model.evaluate(X_train, Y_train, verbose=0)
#print("Error: %.2f%%" % (100-scores[1]*100))



# CV:
from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=N_EPOCH, batch_size=50, verbose=2)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
print("Output: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# test / prediction





# ref http://stackoverflow.com/questions/40046619/keras-tensorflow-gives-the-error-no-attribute-control-flow-ops
#import tensorflow as tf
#tf.python.control_flow_ops = tf


# Result of MLP:
#Train on 60000 samples, validate on 10000 samples
#Epoch 1/10
#12s - loss: 0.2760 - acc: 0.9220 - val_loss: 0.1360 - val_acc: 0.9594
#Epoch 2/10
#13s - loss: 0.1089 - acc: 0.9688 - val_loss: 0.0897 - val_acc: 0.9738
#Epoch 3/10
#13s - loss: 0.0693 - acc: 0.9798 - val_loss: 0.0757 - val_acc: 0.9764
#Epoch 4/10
#12s - loss: 0.0484 - acc: 0.9861 - val_loss: 0.0711 - val_acc: 0.9781
#Epoch 5/10
#13s - loss: 0.0355 - acc: 0.9899 - val_loss: 0.0641 - val_acc: 0.9793
#Epoch 6/10
#12s - loss: 0.0252 - acc: 0.9934 - val_loss: 0.0600 - val_acc: 0.9805
#Epoch 7/10
#13s - loss: 0.0203 - acc: 0.9943 - val_loss: 0.0618 - val_acc: 0.9804
#Epoch 8/10
#13s - loss: 0.0147 - acc: 0.9965 - val_loss: 0.0598 - val_acc: 0.9818
#Epoch 9/10
#12s - loss: 0.0098 - acc: 0.9982 - val_loss: 0.0586 - val_acc: 0.9819
#Epoch 10/10
#14s - loss: 0.0076 - acc: 0.9986 - val_loss: 0.0588 - val_acc: 0.9822
#Baseline Error: 1.78%

# Result of simple CNN:
#Epoch 1/10
#180s - loss: 0.2536 - acc: 0.9268 - val_loss: 0.0892 - val_acc: 0.9736
#Epoch 2/10
#196s - loss: 0.0753 - acc: 0.9775 - val_loss: 0.0491 - val_acc: 0.9841
#Epoch 3/10
#182s - loss: 0.0528 - acc: 0.9838 - val_loss: 0.0460 - val_acc: 0.9862
#Epoch 4/10
#181s - loss: 0.0419 - acc: 0.9868 - val_loss: 0.0400 - val_acc: 0.9886
#Epoch 5/10
#181s - loss: 0.0327 - acc: 0.9891 - val_loss: 0.0354 - val_acc: 0.9892
#Epoch 6/10
#183s - loss: 0.0275 - acc: 0.9914 - val_loss: 0.0395 - val_acc: 0.9876
#Epoch 7/10
#196s - loss: 0.0226 - acc: 0.9928 - val_loss: 0.0336 - val_acc: 0.9888
#Epoch 8/10
#182s - loss: 0.0198 - acc: 0.9936 - val_loss: 0.0389 - val_acc: 0.9881
#Epoch 9/10
#185s - loss: 0.0162 - acc: 0.9946 - val_loss: 0.0367 - val_acc: 0.9889
#Epoch 10/10
#179s - loss: 0.0141 - acc: 0.9954 - val_loss: 0.0358 - val_acc: 0.9894
#Baseline Error: 1.06%

# Result of larger CNN:
#Epoch 1/10
#196s - loss: 0.4050 - acc: 0.8752 - val_loss: 0.0790 - val_acc: 0.9753
#Epoch 2/10
#191s - loss: 0.0950 - acc: 0.9708 - val_loss: 0.0447 - val_acc: 0.9870
#Epoch 3/10
#197s - loss: 0.0658 - acc: 0.9803 - val_loss: 0.0452 - val_acc: 0.9850
#Epoch 4/10
#202s - loss: 0.0533 - acc: 0.9833 - val_loss: 0.0360 - val_acc: 0.9886
#Epoch 5/10
#207s - loss: 0.0458 - acc: 0.9858 - val_loss: 0.0320 - val_acc: 0.9900
#Epoch 6/10
#217s - loss: 0.0426 - acc: 0.9862 - val_loss: 0.0281 - val_acc: 0.9912
#Epoch 7/10
#233s - loss: 0.0368 - acc: 0.9882 - val_loss: 0.0275 - val_acc: 0.9905
#Epoch 8/10
#223s - loss: 0.0334 - acc: 0.9892 - val_loss: 0.0295 - val_acc: 0.9900
#Epoch 9/10
#235s - loss: 0.0298 - acc: 0.9903 - val_loss: 0.0261 - val_acc: 0.9906
#Epoch 10/10
#195s - loss: 0.0275 - acc: 0.9914 - val_loss: 0.0295 - val_acc: 0.9909
#Baseline Error: 0.91%

'''
Result of large CNN /w gpu:
Epoch 1/10
1s - loss: 0.3776 - acc: 0.8797 - val_loss: 0.0809 - val_acc: 0.9744
Epoch 2/10
1s - loss: 0.0921 - acc: 0.9712 - val_loss: 0.0468 - val_acc: 0.9854
Epoch 3/10
1s - loss: 0.0672 - acc: 0.9791 - val_loss: 0.0375 - val_acc: 0.9881
Epoch 4/10
1s - loss: 0.0533 - acc: 0.9831 - val_loss: 0.0325 - val_acc: 0.9886
Epoch 5/10
1s - loss: 0.0462 - acc: 0.9857 - val_loss: 0.0316 - val_acc: 0.9887
Epoch 6/10
1s - loss: 0.0400 - acc: 0.9875 - val_loss: 0.0297 - val_acc: 0.9896
Epoch 7/10
1s - loss: 0.0366 - acc: 0.9881 - val_loss: 0.0236 - val_acc: 0.9922
Epoch 8/10
1s - loss: 0.0335 - acc: 0.9891 - val_loss: 0.0273 - val_acc: 0.9903
Epoch 9/10
1s - loss: 0.0303 - acc: 0.9901 - val_loss: 0.0225 - val_acc: 0.9926
Epoch 10/10
1s - loss: 0.0271 - acc: 0.9913 - val_loss: 0.0243 - val_acc: 0.9917
Baseline Error: 0.83%
'''
