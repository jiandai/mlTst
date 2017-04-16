# -*- coding: utf-8 -*-
"""
Version 20170125 by Jian: follow the tutorial
http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


Version 20170127 by Jian: test on rescomp 
ImportError: cannot import name model_selection
result:
>>> sklearn.__version__
'0.15.2'

Version 20170128 by Jian: test on home laptop
Version 20170212 by Jian: use keras for iris
Version 20170220 by Jian: recap, revisit keras, *packaging
Version 20170414.1 by Jian: fork to iris-tf.py for exercise tf
Version 20170414.2 by Jian: multinomial regression exercise /w iris dataset
Version 20170415.1 by Jian: ref https://www.tensorflow.org/get_started/tflearn
Version 20170415.2 by Jian: ref https://www.tensorflow.org/get_started/input_fn for using batch
Version 20170416 by Jian: ref https://www.tensorflow.org/get_started/monitors for logging
ERRORs:
tf.contrib.learn.metric_spec.MetricSpec(
AttributeError: module 'tensorflow.contrib.learn' has no attribute 'metric_spec'

prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
AttributeError: module 'tensorflow.contrib.learn' has no attribute 'prediction_key'

to-do:
. multiple gpu from start
. equivalent W's


"""

# User tf source
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)
#With INFO-level logging, tf.contrib.learn automatically outputs training-loss metrics to stderr after every 100 steps.

import numpy as np
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

'''
print(training_set.data)
print(training_set.target)
print(test_set.data)
print(test_set.target)
'''
validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics)


feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="./tmp/",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))


def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y


classifier.fit(input_fn=get_train_inputs, steps=5000,
        monitors=[validation_monitor])
#classifier.fit(x=training_set.data, y=training_set.target, steps=1000)

eval_out = classifier.evaluate(input_fn=get_test_inputs, steps=1)
#print(dir(eval_out))
#print(eval_out.keys())
#dict_keys(['accuracy', 'loss', 'global_step', 'auc'])
print (eval_out['accuracy'])
print (eval_out['loss'])
print (eval_out['auc'])

pred = classifier.predict(input_fn= (lambda : np.array( [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)))
#print(type(pred)) #<class 'generator'>
print(list(pred))
quit()








# Use UCI source
#url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
def acquire_data():
	df= pandas.read_csv(url,header=None)
	df.to_csv(csv_path,header=False,index=False)
#acquire_data()

# Prepare 1-hot dataset 
csv_path='../data/iris.csv'
import pandas
df= pandas.read_csv(csv_path,header=None)

X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(Y)
dummy_y = encoder.transform(Y)
#dummy_y = pandas.get_dummies(df.iloc[:,4]).values


# inference
x = tf.placeholder(tf.float32,[None,4])
W = tf.Variable(tf.zeros([4,3]),tf.float32)
b = tf.Variable(tf.zeros([3]),tf.float32)
logits = tf.matmul(x,W)+b
softmax = tf.nn.softmax(logits)
y = tf.placeholder(tf.float32,[None,3])
# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(cross_entropy)
# train
train = tf.train.GradientDescentOptimizer(.1).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(800):
	sess.run(train,feed_dict= {x:X,y:dummy_y})
print(sess.run(loss,feed_dict= {x:X,y:dummy_y}))
pred_prob = sess.run(softmax,feed_dict= {x:X,y:dummy_y})
for j in range(pred_prob.shape[0]):
	print(j,pred_prob[j])
print(sess.run([W]))
print(sess.run([b]))


quit()
#df_1hot = pandas.concat([df.iloc[:,0:4],pandas.get_dummies(df.iloc[:,4])],axis=1)
#df_1hot.to_csv('../data/iris_1hot.csv',header=False)
#print(df_1hot)

#csv_path='../data/iris_1hot.csv'
#import collections
#Dataset = collections.namedtuple('Dataset', ['data', 'target'])
# ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
#iris = tf.contrib.learn.datasets.base.load_csv_without_header(csv_path,features_dtype=np.float,target_dtype=np.int,target_column=range(4,7))
# target_column can only be a single number
#iris = Dataset(data=df.iloc[:,0:4].values, target=pandas.get_dummies(df.iloc[:,4]).values)
#print(iris.target)
#quit()








#dummy_y= tf.one_hot(encoded_Y,3,1,0) 
# TypeError: Value passed to parameter 'indices' has DataType string not in list of allowed values: uint8, int32, int64
# TypeError: The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, or numpy ndarrays.
#Y = df.iloc[:,4].values

#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(df.iloc[:,4].values)
#print encoded_Y

# Not sure how to use OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#encoder.fit(encoded_Y.reshape(-1,1))
#print encoder.transform(encoded_Y.reshape(-1,1))











#### Keras ################################

# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)
print(dummy_y)






from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3,input_dim=4,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,dummy_y)
print(model.summary())


def baseline_model():
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
from sklearn.model_selection import KFold
seed=7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Baseline: 96.67% (4.47%)

