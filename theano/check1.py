# ver <20170214 by jian: benchmark on laptop
# ver 20170214 by jian: benchmark on rescomp, cpu vs gpu
# ref http://deeplearning.net/software/theano/tutorial/using_gpu.html
# ref http://deeplearning.net/software/theano/library/config.html
# ver 20180219 by jian: review for stat presentation
from theano import function, config, shared, tensor, sandbox
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 2 #1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
t0 = time.time()
for i in range(iters):
    r = f() # note r is a numpy.ndarray of shape (230400,)
    print("Result of it [%4d] is {%s}" % (i, r,))
t1 = time.time()

print("Looping %d times took %f seconds" % (iters, t1 - t0))

print(f.maker.fgraph.toposort())
if numpy.any([isinstance(x.op, tensor.Elemwise) and ('Gpu' not in type(x.op).__name__) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')






# My result:
# laptop
#[Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
#Looping 1000 times took 16.188000 seconds
#Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
#  1.62323284]
#Used the cpu

# rescomp cpu
#[Elemwise{exp,no_inplace}(<TensorType(float64, vector)>)]
#Looping 1000 times took 3.679199 seconds
#Result is [ 1.23178032  1.61879341  1.52278065 ...,  2.20771815  2.29967753
#  1.62323285]
#Used the cpu


# rescomp gpu
#Using gpu device 0: Tesla M60 (CNMeM is enabled with initial size: 95.0% of memory, cuDNN 5103)
#/gne/home/daij12/.local/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
#  warnings.warn(warn)
#[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
#Looping 1000 times took 0.264635 seconds
#Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#  1.62323296]
#Used the gpu

# rescomp without loading cuda:
#ERROR (theano.sandbox.cuda): nvcc compiler not found on $PATH. Check your nvcc installation and try again.
#[Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
#Looping 1000 times took 1.971706 seconds
#Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
#  1.62323284]
#Used the cpu
