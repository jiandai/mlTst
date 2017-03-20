# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 06:43:45 2017

@author: daij12


ver 20170129 by jian: adjust to home laptop and source from web
ref: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
ver 20170212 by jian: recap
ref: http://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/
"""

import numpy
seed=7
numpy.random.seed(seed)

dataset=numpy.loadtxt('./data/pima-indians-diabetes.csv',delimiter=',')
print(dataset.shape)
X=dataset[:,0:8]
Y=dataset[:,8]
print(X.shape)
print(Y.shape)



from keras.models import Sequential
from keras.layers import Dense

# Step 1 def the model
model=Sequential()
model.add(Dense(12,input_dim=8,init='uniform',activation='relu'))
#model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))
# 'linear'/'mse','sigmoid'/'binary_crossentropy','softmax'/'categorical_crossentropy'

# Step 2 compile the model
# 'sgd','adam',rmsprop'
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Step 3 fit the model
#model.fit(X,Y,nb_epoch=150,batch_size=10)
#model.fit(X,Y,nb_epoch=150,batch_size=10,verbose=0)
history = model.fit(X,Y,nb_epoch=150,batch_size=10,verbose=2)

# Step 4 evaluate the model
print (model.evaluate(X,Y))
loss,accuracy=model.evaluate(X,Y,verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], accuracy*100))
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# Step 5 make prediction


predictions=model.predict(X)
print(predictions)
rounded=[round(i) for i in predictions.flatten()]
print(rounded)

probabilities = model.predict(X)
print(probabilities.shape)
predictions = [float(round(x)) for x in probabilities.flatten()]
accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

quit()

#%%
import pandas
#%%
url='http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataframe= pandas.read_csv(url,header=None)
dataset= dataframe.values








#%%

model_json = model.to_json()
with open('model.json','w') as json_file:
	json_file.write(model_json)

model_yaml = model.to_yaml()
with open('model.yaml','w') as yaml_file:
	yaml_file.write(model_yaml)

model.save_weights('model.h5')

#%%

