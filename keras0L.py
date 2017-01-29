# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 06:43:45 2017

@author: daij12

ref: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
ver 20170129 by jian: adjust to home laptop and source from web
"""


#%%
import pandas
#%%
url='http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe= pandas.read_csv(url,header=None)
dataset= dataframe.values


#%%
import numpy
#%%
#dataset=numpy.loadtxt('C:/gitLocal/ML/pima-indians-diabetes.csv',delimiter=',')
#%%
X=dataset[:,0:8]
Y=dataset[:,8]


#%%
seed=7
numpy.random.seed(seed)


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
