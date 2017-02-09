# -*- coding: utf-8 -*-
"""
Version 20170129 by Jian: issue /w sklearn version
>>> sklearn.__version__
'0.15.2'
Version 20170208 by Jian: install sklearn locally
pip install --user -U scikit-learn
Version 20170209 by Jian: use semeion data as another example
"""
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')

import pandas as pd
dataset = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data',sep='\s+',header=None,engine='python')

# col 0-255: data
# col 256-265: one-hot label


#print dataset.shape

X,Y = dataset[range(256)],dataset[range(256,266)]
#print X.shape
#print Y.shape

#import matplotlib.pyplot as plt
# Convert the 1st row to an array to visualize
#print type(X.iloc[15].values.reshape((16,16),order='C'))
#plt.imshow(X.iloc[15].values.reshape((16,16),order='C'),cmap='hot')
#plt.show()
#plt.imshow(X.iloc[16].values.reshape((16,16)),cmap='gray')
#plt.show()
#plt.imshow(X.iloc[17].values.reshape((16,16)))
#plt.show()
#plt.imshow(X.iloc[18].values.reshape((16,16)))
#plt.show()

#print Y.sum(axis=1)

y=[]
for r in range(Y.shape[0]):
	for c in range(256,266):
		if Y[c].iloc[r]==1:
			#print r,c-256
			y.append(c-256)
import numpy as np
y=np.array(y)

seed=7
from sklearn import model_selection
X_tr,X_tt,y_tr,y_tt = model_selection.train_test_split(X,y,test_size=.3,random_state=seed)

print X_tr.shape
print X_tt.shape
print y_tr.shape
print y_tt.shape


from sklearn import metrics

def algoTst(model):
	model.fit(X_tr,y_tr)
	yhat_tt=model.predict(X_tt)
	print metrics.accuracy_score(y_tt, yhat_tt)
	print metrics.confusion_matrix(y_tt, yhat_tt)
	print metrics.classification_report(y_tt, yhat_tt)

def cvTst(model,FOLD=3):
	seed=11
	kfold=model_selection.KFold(n_splits=FOLD,random_state=seed)
	cv_results = model_selection.cross_val_score(model,X_tr,y_tr,cv=kfold,scoring='accuracy')
	print cv_results

from sklearn import linear_model
algoTst(model = linear_model.LogisticRegression())
cvTst(model = linear_model.LogisticRegression())


from sklearn import svm
algoTst(model = svm.SVC())
cvTst(model = svm.SVC())


from sklearn import neighbors
algoTst(model = neighbors.KNeighborsClassifier())
cvTst(model = neighbors.KNeighborsClassifier())

from sklearn import tree
algoTst(model = tree.DecisionTreeClassifier())
cvTst(model = tree.DecisionTreeClassifier())

from sklearn import naive_bayes
algoTst(model = naive_bayes.GaussianNB())
cvTst(model = naive_bayes.GaussianNB())

from sklearn import discriminant_analysis
algoTst(model = discriminant_analysis.LinearDiscriminantAnalysis())
cvTst(model = discriminant_analysis.LinearDiscriminantAnalysis())


