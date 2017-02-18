# coding: utf-8
'''
version <20170209 by Jian: load iris from sklearn
version 20170209 by Jian: play /w algo
ref: http://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
version 20170216 by Jian: more on sklearn

'''

from sklearn import datasets
dataset = datasets.load_iris()

# fit a CART model to the data
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
print(model)

model.fit(dataset.data, dataset.target)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



# manipulate the dataset
iris = dataset
print(iris.keys())

# create a Iris data frame
import pandas as pd
irisDF=pd.concat([pd.DataFrame(iris.data,columns=iris.feature_names),pd.DataFrame(iris.target,columns=['SpeciesCode'])],axis=1) 

irisDF['Species']=iris.target_names [irisDF.SpeciesCode]

print(irisDF.head())

