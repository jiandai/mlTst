"""
Version 20170129 by Jian: issue /w sklearn version (0.15.2)
Version 20170208 by Jian: install sklearn locally
pip install --user -U scikit-learn
Version 20170209 by Jian: use semeion data as another example
Version 20170504 by Jian: revamp, companion with ipynb file, hand code CNN
"""


# semeion data is downloaned, named as "semeion.data.txt", and put in the following path relative to the current working dir
import pandas as pd
#dataset = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data',sep='\s+',header=None,engine='python')
#dataset = pd.read_table('data/semeion.data.txt',sep='\s+',header=None,engine='python')
dataset = pd.read_table('./data/semeion.data.txt',sep='\s+',header=None)
# col 0-255: data
# col 256-265: one-hot label
X,Y = dataset[list(range(256))],dataset[list(range(256,266))] # Seperate features and labels

print(X.shape, Y.shape)

x = X.values.reshape(X.shape[0],16,16)
y = Y.values

import numpy as np
import math
def conv2(phi, K,
          #padding = 5, # test
          #padding = 10, # SAME
          padding = 0, # VALID
          stride=2
         ):
    size = phi.shape[0] # Assume X is a square
    kernel_size = K.shape[0]
    # Key formula
    out_size = (size+2*padding-kernel_size)/stride+1
    out_size = int(out_size)
    #print('Output size:',out_size)
    padded_x = np.zeros([size+2*padding,size+2*padding])
    padded_x[padding:(padding+size),padding:(padding+size)] = phi
    psi = np.zeros([out_size,out_size])
    for i in range(out_size):
        for j in range(out_size):
            x0,y0=i*stride,j*stride
            #print(x0,y0)
            # Compute the convolution for output at [i,j]
            for a in range(kernel_size):
                for b in range(kernel_size):
                    psi[i,j] += K[a,b]*padded_x[x0+a,y0+b]
    return psi

# A simply convolutional neural network
#1*16*16 => 3*7*7 by using zero padding, 4*4 kernel, and stride 2
TR_SIZE = 100000
Check =1000
shuffled_indice = np.random.choice(x.shape[0], size=TR_SIZE, replace=True)
x=x[shuffled_indice]
y=y[shuffled_indice]
num_channel = 3
K_SIZE = 4
K = np.random.random([num_channel,K_SIZE,K_SIZE])
H_SIZE = 7
h = np.zeros([num_channel,H_SIZE,H_SIZE])
W = np.random.normal(0,.01,size=[10,num_channel,H_SIZE,H_SIZE])
b = np.zeros(10)
s = 2
grad_W = np.zeros(W.shape)
grad_b = np.zeros(b.shape)
grad_K = np.zeros(K.shape)
lr = .1
X_ = np.zeros([7,7,4,4])
for id in range(x.shape[0]):
    x_ = x[id]
    y_ = y[id]
    for ch in range(num_channel):
        h[ch] = conv2(x_,K[ch])
    h = 1/(1+np.exp(-h))
    logits = np.tensordot(h,W,axes=([0,1,2],[1,2,3])) + b
    p = np.exp(logits) / np.exp(logits).sum()
    cross_entropy = -(y_*np.log(p)).sum()/math.log(10) # aka normalized log likelihood, Kullback-Lebler divergent
    if id % Check == 0:
        print(id,cross_entropy)
    grad_W = - np.tensordot((y_ - p).reshape(10,1), h.reshape(1,3,7,7),  axes=([1],[0]))  /math.log(10)
    grad_b = - np.tensordot((y_ - p).reshape(10,1), np.ones([1]),  axes=([1],[0]))  /math.log(10)    #    
    for i in range(7):
        for j in range(7):
            for l in range(4):
                for m in range(4):
                    X_[i,j,l,m] = x_[s*i+l,s*j+m]
    grad_K = - np.tensordot(np.tensordot((y_ - p), W,  axes=([0],[0])) * h * (1-h), X_,axes=([1,2],[0,1]))  /math.log(10)
    W += -grad_W * lr
    b += -grad_b * lr
    K += -grad_K * lr
#



quit()
import os
import sys
sys.path.insert(0, os.path.expanduser('~')+'/.local/lib/python2.7/site-packages')




#import matplotlib.pyplot as plt
#plt.imshow(X.iloc[19].values.reshape((16,16),order='C'),cmap='hot')
#plt.show()
#plt.imshow(X.iloc[20].values.reshape((16,16)),cmap='gray')
#plt.show()
#plt.imshow(X.iloc[21].values.reshape((16,16)))
#plt.show()
#plt.imshow(X.iloc[22].values.reshape((16,16)))
#plt.show()

#print Y.sum(axis=1)


# convert 1-hot code to single variable
y=[]
for r in range(Y.shape[0]):
	for c in range(256,266):
		if Y[c].iloc[r]==1:
			#print r,c-256
			y.append(c-256)
y=np.array(y)

seed=7
from sklearn import model_selection
X_tr,X_tt,y_tr,y_tt = model_selection.train_test_split(X,y,test_size=.3,random_state=seed)

print( X_tr.shape)
print( X_tt.shape)
print( y_tr.shape)
print( y_tt.shape)


from sklearn import metrics

def algoTst(model):
	model.fit(X_tr,y_tr)
	yhat_tt=model.predict(X_tt)
	print(metrics.accuracy_score(y_tt, yhat_tt))
	print(metrics.confusion_matrix(y_tt, yhat_tt))
	print(metrics.classification_report(y_tt, yhat_tt))

def cvTst(model,FOLD=3):
	seed=11
	kfold=model_selection.KFold(n_splits=FOLD,random_state=seed)
	cv_results = model_selection.cross_val_score(model,X_tr,y_tr,cv=kfold,scoring='accuracy')
	print(cv_results)

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


