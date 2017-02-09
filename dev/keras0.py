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
from keras.models import Sequential
#%%
from keras.layers import Dense


#%%
model=Sequential()
model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

#%%
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#%%
#model.fit(X,Y,nb_epoch=150,batch_size=10)
#model.fit(X,Y,nb_epoch=150,batch_size=10,verbose=0)
model.fit(X,Y,nb_epoch=150,batch_size=10,verbose=2)
#%%
#scores=model.evaluate(X,Y)
scores=model.evaluate(X,Y,verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#%%
predictions=model.predict(X)
#%%
rounded=[round(i) for i in predictions.flatten()]
print(rounded)



#%%

model_json = model.to_json()
with open('model.json','w') as json_file:
	json_file.write(model_json)

model_yaml = model.to_yaml()
with open('model.yaml','w') as yaml_file:
	yaml_file.write(model_yaml)

model.save_weights('model.h5')

#%%

