"""
ver <20170211 by jian:
ver 20170211 by jian:
	ref http://deeplearning.net/tutorial/lenet.html
"""
#%%
import numpy
rng = numpy.random.RandomState(7)
#%
from theano import tensor as T
#%%
input = T.tensor4(name='input')
#%%
w_shp = (2,3,9,9) # output channel, input channel, size
w_bound = numpy.sqrt(3*9*9)
import theano
W = theano.shared (numpy.asarray(rng.uniform(low=-1.0/w_bound,high=1.9/w_bound,size=w_shp),dtype=input.dtype),name='W')
#%%
b_shp = (2,)
b = theano.shared(numpy.asarray(rng.uniform(low=-.5,high=.5,size=b_shp),dtype=input.dtype),name='b')
#%%
from theano.tensor.nnet import conv2d
conv_out = conv2d (input, W)
#%%
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))
#%%
f = theano.function([input],output)

#%%
from PIL import Image
#%%
#img = Image.open('C:\\gitLocal\\ML\\data\\3wolfmoon.jpg')
img = Image.open('./data/3wolfmoon.jpg') # assume to run the script from proj root
#%%
img = numpy.asarray(img,dtype='float64') / 256.
#%%
img_ = img.transpose(2,0,1).reshape(1,3,639,516)
#%%
# Apparently img_.dtype is 'float64'
filtered_img = f(img_.astype('float32'))
print type(filtered_img)
print filtered_img.shape
quit()
#%%
'''
import pylab
#%%
pylab.subplot(1,3,1)
pylab.axis('off')
pylab.imshow(img)
pylab.gray()
pylab.subplot(1,3,2)
pylab.axis('off')
pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1,3,3)
pylab.axis('off')
pylab.imshow(filtered_img[0,1,:,:])
pylab.show()
'''
#%%
invals = numpy.random.RandomState(1).rand(3,2,5,5)
#%%
input =T.dtensor4('input')
maxpool_shape=(2,2)
from theano.tensor.signal import pool
#%%
pool_out = pool.pool_2d(input, maxpool_shape,ignore_border=True)
f = theano.function([input],pool_out)
print(invals[0,0,:,:])
print(f(invals)[0,0,:,:])
#%%
pool_out = pool.pool_2d(input, maxpool_shape,ignore_border=False)
f = theano.function([input],pool_out)
print(invals[0,0,:,:])
print(f(invals)[0,0,:,:])
#%%
