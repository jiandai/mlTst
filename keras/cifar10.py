# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:01:51 2017

@author: daij12

ref:
    http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
ver 20170308 by jian: to run on server /w 5 and 25 epoches
ver 20180204 by jian: recap, tensorboard
note: 
to use newer version of tb on the server: 
python3 /gne/home/daij12/.local/lib/python3.6/site-packages/tensorflow/tensorboard/tensorboard.py --logdir log/
to-do:
embedding layer in tf/keras, and embeding visualization
"""


# Constant
seed=7
epochs = 20000
#epochs = 25
lrate = 0.01
decay = lrate/epochs

debug_cutoff=32*20

# Packages
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.datasets import cifar10



# Logic:
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
#Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#170500096/170498071 [==============================] - 107s     
#Untaring file...


#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#(50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)



#%%
#from matplotlib import pyplot
#from scipy.misc import toimage
#%%
#for i in range(0, 9):
#	pyplot.subplot(330 + 1 + i)
#	pyplot.imshow(toimage(X_train[i]))
#pyplot.show()


numpy.random.seed(seed)

#import theano
#K.set_image_dim_ordering('th')
#input_shape = (3, 32, 32)
'''
if X_train.shape==(50000,32,32,3):
	X_train=numpy.swapaxes(numpy.swapaxes(X_train,2,3),1,2)
if X_test.shape==(10000,32,32,3):
	X_test=numpy.swapaxes(numpy.swapaxes(X_test,2,3),1,2)
'''

K.set_image_dim_ordering('tf')
input_shape = (32, 32, 3)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



X_train = X_train[:debug_cutoff]
y_train = y_train[:debug_cutoff]
X_test = X_test[:debug_cutoff]
y_test = y_test[:debug_cutoff]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



#%%
# one-hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#%%
num_classes=y_test.shape[1]



#%%
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=input_shape, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# https://stackoverflow.com/questions/45265436/keras-save-image-embedding-of-the-mnist-data-set
embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))


sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

tb = TensorBoard(log_dir='./log/', histogram_freq=10, write_graph=True, write_images=True, batch_size=32, write_grads=True,
	embeddings_freq=10, embeddings_layer_names=embedding_layer_names, embeddings_metadata=None)
"""
v 2.0.2 interface in "C:\Anaconda3\lib\site-packages\keras\callbacks.py":
self.log_dir = log_dir
self.histogram_freq = histogram_freq
self.merged = None
self.write_graph = write_graph
self.write_images = write_images
"""



model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32, callbacks=[tb], )
#AssertionError: AbstractConv2d Theano optimization failed: there is no implementation available supporting the requested options. Did you exclude both "conv_dnn" and "conv_gemm" from the optimizer? If on GPU, is cuDNN available and does the GPU support it? If on CPU, do you have a BLAS library installed Theano can link against?
# 9:05am--12:47pm


#%%
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Accuracy: 70.23%
#%%

#%%
# save the model
#model_json = model.to_json()
#with open('cifar10-model.json','w') as json_file:
#        json_file.write(model_json)

#model.save_weights('cifar10-model.h5')

#%%
# load the model
#from keras.models import model_from_json
#with open('cifar10-model.json','r') as json_file:
#        loaded_model_json = json_file.read()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights('cifar10-model.h5')
#loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#score = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
