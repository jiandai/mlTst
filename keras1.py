# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:52:44 2017
@author: daij12

ref: http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
ver 20170210 by jian: linear model vs keras

"""


# Part 0:
import pandas
dataframe = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', delim_whitespace=True, header=None)
dataset = dataframe.values
X = dataset[:,0:13]
Y = dataset[:,13]





# Part 1:


# force to use local packages, in particular, sklearn
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages') 

from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


import matplotlib.pyplot as plt

def lm(model,FOLD=7,seed=11,score='neg_mean_squared_error'):
	# part 0
	kfold = KFold(n_splits=FOLD, random_state=seed)
	# part 1
	results = cross_val_score(model, X, Y, cv=kfold,scoring=score)
	print(results)
	print("%s mean (sd): %.2f (%.2f)" % (score, results.mean(), results.std()))
	# part 2
	Yhat = cross_val_predict(model, X, Y, cv=kfold)
	fig, ax = plt.subplots()
	ax.scatter(Y,Yhat)
	ax.plot([Y.min(),Y.max()],[Yhat.min(),Yhat.max()])
	ax.set_xlabel('Actual')
	ax.set_ylabel('Predicted')
	plt.show()

print('OLS:')
lm(model = linear_model.LinearRegression())
print('Ridge:')
lm(model = linear_model.Ridge(alpha = 0.1))
print('Lasso:')
lm(model = linear_model.Lasso(alpha = 0.1))

#['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def baseline_model():
	model = Sequential()
	model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=10, verbose=0) 



print('1-nnet:')
lm(model = estimator)

#OLS:
#[ -11.30795492  -10.94743251  -32.12487323  -33.57836799  -10.54825524
# -149.57087382  -12.93285837]
#neg_mean_squared_error mean (sd): -37.29 (46.79)
#Ridge:
#[ -11.24378566  -10.66490322  -32.29513882  -33.16572359  -10.3771499
# -149.91511585  -13.06506445]
#neg_mean_squared_error mean (sd): -37.25 (46.94)
#Lasso:
#[ -10.63704843   -8.546795    -37.15196607  -29.66568057  -10.36678128
# -145.68350018  -16.06195239]
#neg_mean_squared_error mean (sd): -36.87 (45.54)
#Using Theano backend.
#1-nnet:
#[-17.17150178  -9.13191837 -45.81825987 -46.42486771 -15.72822738
# -69.06089044 -20.33876135]
#neg_mean_squared_error mean (sd): -31.95 (20.41)






quit()
# Part 2:





import numpy
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
