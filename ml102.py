# -*- coding: utf-8 -*-
"""
Version 20170129 by Jian: issue /w sklearn version
ImportError: cannot import name model_selection
result:
>>> sklearn.__version__
'0.15.2'
Version 20170208 by Jian: install sklearn locally
pip install --user -U scikit-learn
"""
import os
import sys
sys.path.insert(1, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')


#%%
import pandas
#%%
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#%%
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.groupby('class').size())
#%%
#%%
import matplotlib.pyplot as plt
#%%
# cannot run on rescomp
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
#%%
# cannot run on rescomp
#dataset.hist()
#plt.show()
#%%
# cannot run on rescomp
#pandas.tools.plotting.scatter_matrix(dataset)



#%%
#import sklearn
# cannot run on rescomp: ImportError: cannot import name model_selection
from sklearn import model_selection
from sklearn import linear_model
# cannot run on rescomp: ImportError: cannot import name discriminant_analysis
from sklearn import discriminant_analysis
from sklearn import neighbors
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import metrics
#%%
array = dataset.values
#%%
X = array[:,0:4]
Y = array[:,4]
#%%
validation_size = 0.20
#%%
seed = 7
quit()




#%%
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#%%
seed = 7
scoring = 'accuracy'
#%%
models = []
models.append(('LR', linear_model.LogisticRegression()))
models.append(('LDA', discriminant_analysis.LinearDiscriminantAnalysis()))
models.append(('KNN', neighbors.KNeighborsClassifier()))
models.append(('CART',tree.DecisionTreeClassifier()))
models.append(('NB', naive_bayes.GaussianNB()))
models.append(('SVM',svm.SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
 
 
#%%
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#%%
knn = neighbors.KNeighborsClassifier()
#%%
knn.fit(X_train, Y_train)
#%%
predictions = knn.predict(X_validation)
#%%
metrics.accuracy_score(Y_validation, predictions)
#%%
metrics.confusion_matrix(Y_validation, predictions)
#%%
print(metrics.classification_report(Y_validation, predictions))
