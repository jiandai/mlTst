# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:52:44 2017
@author: daij12

ref: http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
ver 20170210 by jian: recap

"""



import pandas
dataframe = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', delim_whitespace=True, header=None)
print(type(dataframe))
print(dataframe.shape)
dataset = dataframe.values
print(type(dataset))
print(dataset.shape)

X = dataset[:,0:13]
Y = dataset[:,13]


from keras.models import Sequential
from keras.layers import Dense

def baseline_model():
	model = Sequential()
	model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


seed = 7
import numpy
numpy.random.seed(seed)
from keras.wrappers.scikit_learn import KerasRegressor
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0) 




# force to use local packages, in particular, sklearn
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages') 
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=seed)

from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X, Y, cv=kfold) # Error on home laptop

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Results: 38.04 (28.15) MSE






numpy.random.seed(seed)
estimators = []
from sklearn.preprocessing import StandardScaler
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)))

from sklearn.pipeline import Pipeline
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Standardized: 28.24 (26.25) MSE





def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
	model.add(Dense(6, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Larger: 24.60 (25.65) MSE

#%%
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
 
#%%
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Wider: 21.64 (23.75) MSE
