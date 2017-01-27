# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 06:43:45 2017

@author: daij12

ref: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""

#%%
import numpy
#%%
seed=7
numpy.random.seed(seed)
#%%
dataset=numpy.loadtxt('C:/Users/daij12/Documents/Analysis/ml/Coldz/pima-indians-diabetes.csv',delimiter=',')
#%%
X=dataset[:,0:8]
Y=dataset[:,8]

#%%

from keras.models import model_from_json
with open('model.json','r') as json_file:
	loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#%%
from keras.models import model_from_yaml
with open('model.yaml','r') as yaml_file:
	loaded_model_yaml = yaml_file.read()
loaded_model = model_from_yaml(loaded_model_yaml)
loaded_model.load_weights('model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
