"""
Stochastic Fisher's Scoring as a logistic regression solver
# Dev diary and authors:
- 20180630 : first pass
- 20180701 : 
 - use validation
 - add bias
 - test convergence conditions: norm of gradients gives overfitting and in stochastic trainng, norm convergence does not mean much

# Issue:

# To-do:

"""
import numpy as np
import tensorflow as tf

def hessian(y, x):
  """
  # assume y is a scalar
  # assume x is a vector (1d tensor)
  # note the tf implementation is tf.hessians
  """
  assert len(y.shape)==0 and len(x.shape)==1
  D = x.shape[0]
  dy = tf.gradients(y, x) [0]
  return tf.concat([tf.gradients(dy[i], x) for i in range(D)], axis=0)

# Optimizer
def score(loss, W, lr):
  """ Newton-Raphson solver
  """
  fisher_score = tf.gradients(loss, W) #
  hess = tf.hessians(loss, W) #
  inv_h = tf.matrix_inverse(hess)

  # diagnosis:
  #det = tf.reshape(tf.matrix_determinant(hess),[])
  norm = tf.norm(fisher_score)

  # Update
  delta_W = tf.tensordot(inv_h[0], fisher_score[0], axes=[[1],[0]])
  return W.assign_sub(lr * delta_W), norm

# Generator class
class DataSim:
  """ generator of simulated data of a logistic reg model
  """
  def __init__(self,true_weight,x_dim,batch_size=16, pos_cls_frac=.5, std=1.5):
    self.batch_size = batch_size
    self.n1 = round(batch_size * pos_cls_frac)
    self.n0 = batch_size - self.n1
    self.D = x_dim
    self.W = true_weight
    self.scale = std

  def gen(self):
    while True:
      x1 = np.random.normal(loc=self.W+.5, size=(self.n1,self.D), scale=self.scale)
      x0 = np.random.normal(loc=-self.W+.5, size=(self.n0,self.D), scale=self.scale)
      y1 = np.array([1.]*self.n1)
      y0 = np.array([0.]*self.n0)
      x = np.vstack( (x0,x1) )
      y = np.vstack( (y0,y1) )
      #y = np.random.binomial(1, p, p.shape[0])
      #y = np.round(p)
      y = y.reshape(self.batch_size, 1)
      yield x, y



# hyperparameters:
n_iter = 100
x_dim=3
mini_batch_size = 1024*4
learning_rate = .110

# data generator
W0 = np.array([.7] * x_dim + [0.], dtype=np.float32) 
data_sim = DataSim(W0[:-1], x_dim, batch_size=mini_batch_size)
data_gen = data_sim.gen()



# The Graph:
X=tf.placeholder(dtype=tf.float32, shape=[None, x_dim])
Y=tf.placeholder(dtype=tf.float32, shape=[None,1])
W = tf.get_variable("Weight", [x_dim+1], initializer=tf.random_normal_initializer(), dtype=tf.float32)

logit = tf.tensordot(W[:-1], X, [[0],[1]]) +W[-1]
logit = tf.reshape(logit, [-1,1])
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit) ) # better protected

#p_hat = 1/(1+tf.exp (-1.*logit))
#p_hat = tf.sigmoid(logit)
#loss = -tf.reduce_mean(Y*tf.log(p_hat) + (1-Y)*tf.log (1-p_hat)) # unsafe

score_op = score(loss, W, learning_rate)

# SDG:
#delta_W = -learning_rate * fisher_score[0]
## Compute the ``norm`` of delta_W
#update_op = tf.cond(tf.abs(det)>1e-5, lambda : W.assign_add(delta_W), lambda : W)


# Run the graph
sess = tf.Session()
sess.run(tf.variables_initializer(tf.global_variables()))


# train:
min_v_loss=100.
min_v_loss_idx=-1

val_x,val_y = next(data_gen)
for j in range(n_iter):
  x,y = next(data_gen)
  #_w, _delta_w,_det,_norm, _loss = sess.run([update_op, delta_W, det, norm, loss], feed_dict={X:x, Y:y})
  _update, _loss, _w =sess.run([score_op, loss, W], feed_dict={X:x, Y:y})
  _w, _norm = _update
  v_loss = sess.run(loss, feed_dict={X:val_x, Y:val_y})
  print('Iter %d:' % j, _w, _loss, v_loss, _norm)
  #print('Iter %d:' % j, _w, _det, _norm, delta_w_norm, _loss, v_loss)
  if v_loss<min_v_loss:
    min_v_loss = v_loss; min_v_loss_idx = j

# the best weight
print(min_v_loss_idx, min_v_loss)


