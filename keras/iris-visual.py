"""
Version 20170125 by Jian: follow the tutorial
http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


Version 20170127 by Jian: test on server
ImportError: cannot import name model_selection
due to
>>> sklearn.__version__
'0.15.2'

Version 20170128 by Jian: test on home laptop
Version 20170212 by Jian: use keras for iris
Version 20170220 by Jian: recap, revisit keras, *packaging
Version 20170304 by Jian: review CV
Version 20170317 by Jian: test multiple gpu => Not sure whether it works
Version 20170402 by Jian: rerun without turning on >1 gpu
Version 20170404 by Jian: rerun turning on >1 gpu
Version 20170407 by Jian: fork from iris-keras.py to test visual https://github.com/fchollet/hualos
"""


csv_path='../data/iris.csv'
import pandas
df= pandas.read_csv(csv_path,header=None)
dataset = df.values # return a <class 'numpy.ndarray'> type
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(encoded_Y)

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
remote = callbacks.RemoteMonitor(root='http://localhost:9000')
# baseline test : multinomial regression
model = Sequential()
model.add(Dense(3,input_dim=4,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,dummy_y,nb_epoch=1000,batch_size=20,validation_split=0.3,callbacks=[remote])
print(model.summary())
print(model.predict(X))


