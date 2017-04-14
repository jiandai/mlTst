# ref https://www.tensorflow.org/get_started/get_started
import tensorflow as tf
import numpy as np

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=3,
                                              num_epochs=300)

#estimator.fit(input_fn=input_fn, steps=32)
estimator.fit(input_fn=input_fn)

print (estimator.evaluate(input_fn=input_fn))
