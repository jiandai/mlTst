""" This script trains a Lenet-like cnn to predict on UCI Semeion dataset

Summary:
- Minimal/partial (re)implementation of nn.Module and nn.optim in pytorch
- Data feed pipeline development
- GPU computing

Version and authors:
- 20180617 by Jian: 
 -- code lenet from bottom up
 -- build data pipe including training/validation split
 -- implement ``to``method and train on gpu

To-do:
- code minibatch /w size>1
- code cross entropy
- optimization gpu run

Issue:
- currently gpu does not perform better than cpu (?)

Note:
- effect of max pool by 2 on odd dimension is to floor the quotient

Ref: 
https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.names
"""

# package
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# class / func / utilities
class Lenet:
  """ Minimal / partial (re-) implementation of nn.Module
  """
  def __init__(self, input_size,ch, ks, pl,nu, debug=False):
    """
      ch: number of conv2d channels
      ks: kernal size of conv2d (use square kernel only)
      pl: 2d pooling size
      nu: number of fn units
    """
    self.num_of_conv_layers = len(ks)
    assert len(ch) ==self.num_of_conv_layers+1
    assert len(pl) ==self.num_of_conv_layers

    layers=[]
    r, c = input_size
    if debug: print("r=", r, ", c=",c)
    for j, cc in enumerate(zip(ch[:self.num_of_conv_layers], ch[1:], ks, pl)):
      layer = 'conv'+str(j+1)
      setattr(self, layer, nn.Conv2d(cc[0], cc[1], cc[2]))
      r, c = (r+1-cc[2])//cc[3], (c+1-cc[2])//cc[3] # pool output dim
      if debug: print("r=", r, ", c=",c)
      layers.append(layer)
    self.dim_to_flatten = r*c*ch[-1]
    if debug: print("~~~~~~~~~~~~~~~~~~ Flatten (", r, ",", c, ",", ch[-1], ") to ", self.dim_to_flatten, " nodes")
    self.num_of_fn_layers = len(nu)
    nu = [self.dim_to_flatten] + nu
    for j, fd in enumerate(zip(nu[:self.num_of_fn_layers], nu[1:])):
      layer = 'fn'+str(j+1)
      setattr(self, layer, nn.Linear(fd[0], fd[1]))
      layers.append(layer)
    self.pl = pl
    self.layers = layers


  def forward(self,x):
    h = x
    for j in range(self.num_of_conv_layers):
      h = getattr(self, 'conv' + str(j+1)) (h)
      h = F.relu(h) # no padding by default
      h = F.max_pool2d(h, self.pl[0])
    h = h.view(-1, self.dim_to_flatten)
    for j in range(self.num_of_fn_layers):
      h = getattr(self, 'fn' + str(j+1)) (h)
      if j < self.num_of_fn_layers-1:
        h = F.relu(h)
    return h

  def parameters(self):
    for l in self.layers:
      layer = getattr(self, l)
      yield layer.weight
      yield layer.bias

  def to(self, device):
    for l in self.layers:
      layer = getattr(self, l)
      layer.to(device)

  ###
  def zero_grad(self):
    for l in self.layers:
      getattr(self, l).zero_grad()



class Optim:
  def __init__ (self, net, learning_rate):
    self.net = net
    self.learning_rate = learning_rate

  def zero_grad(self):
    self.net.zero_grad()


  def update_parameters(self):
    """ Vanilla SGD
    """
    for p in self.net.parameters():
      p.data -= self.learning_rate * p.grad.data


# Program logic
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
## set up
net = Lenet(input_size= (16, 16),ch=[1,16,32], ks=[3] * 2, pl=[2] * 2,nu=[64,32, 10], debug=False)
net.to(device)


criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
optimizer = Optim(net, learning_rate = .1)

n_iter = 70000 
split = .8
validation_freq = 1000

# Data pipe
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data', delim_whitespace=True, header=None)
# note: col 0 - 255 image data, col 256 - 265 one-hot label for digit 0-9
N = df.shape[0]

# shuffle the data
shuffled_idx = np.random.permutation(N)
df = df.iloc[shuffled_idx]

X, Y = df.iloc[:,:256].values, df.iloc[:,256:].values
X = X.astype(np.float32)
Y = Y.astype(np.float32)
X = X.reshape( (X.shape[0], 1, 16,16) )
X, Y = torch.from_numpy(X), torch.from_numpy(Y)

training_size=round(N * split)
validation_size = N - training_size

training_X, training_Y = X[:training_size], Y[:training_size]
validation_X, validation_Y = X[training_size:], Y[training_size:]


sample_idx = np.random.choice( np.arange(training_size), n_iter)
# minibatch size = 1
for i in range(n_iter): 
  start_time = time.time()
  idx = sample_idx[i] 
  # clear grad buffer
  optimizer.zero_grad()
  
  # generate model input
  y = training_Y[idx].unsqueeze(0)
  x = training_X[idx].unsqueeze(0)
  x = x.to(device)
  y = y.to(device)

  # compute model output
  yhat = net.forward(x)

  # compute loss
  loss = criterion(yhat, y)

  # compute grad of loss w.r.t. par
  loss.backward()

  # update par
  optimizer.update_parameters()


  # validation
  with torch.no_grad():
    if (i+1) % validation_freq== 0:
      val_loss = 0
      for j in range(validation_size):
        x = validation_X[j].unsqueeze(0)
        y = validation_Y[j].unsqueeze(0)
        x = x.to(device)
        y = y.to(device)
        yhat = net.forward(x)
        val_loss += criterion(yhat, y)
      val_loss /= validation_size
      elapsed_time = time.time() - start_time
      print('[{0:8d}] : training loss = [{1: .6f}], validation loss = [{2: .6f}], epoch duration = [{3: .3f}] sec'.format(i+1, loss.data.item(), val_loss.data.item(), elapsed_time))





