'''
resnet50_weights_tf_dim_ordering_tf_kernels.h5
vgg16_weights_tf_dim_ordering_tf_kernels.h5
xception_weights_tf_dim_ordering_tf_kernels.h5
'''
from scipy.misc import imsave, imread, imresize
seed_img = imread('Tiger-cat-full.jpg')
seed_img = imresize(seed_img, (224,224,3))
print(seed_img.shape)
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=True)
print(model.summary())


# cannot be used /w vis
#from keras import backend as K
#K.set_learning_phase(0)


from vis.visualization import visualize_activation #,visualize_activation_with_losses
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm



from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.modifiers import Jitter
from vis.optimizer import Optimizer

from vis.callbacks import GifGenerator
#from vis.utils.vggnet import VGG16

# Build the VGG16 network with ImageNet weights
print('Model loaded.')
layer_name = 'predictions'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print(layer_dict)
for j,l in enumerate(model.layers):
	print(j,l.name)
	if l.name==layer_name:
		I = j
from keras import activations
import vis
#model.layers[I].activation = activations.linear
#model = vis.utils.utils.apply_modifications(model)
# The above statement doesn't work

#img = visualize_activation(model, 33,input_range=(0., 1.), verbose=True, act_max_weight=1,tv_weight=10, lp_norm_weight=20, seed_input = seed_fig, max_iter=1)


# The name of the layer we want to visualize
# (see model definition in vggnet.py)
# https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
#output_class = [20] # "20": ["n01601694", "water_ouzel"]
output_class = [282] # "282": ["n02123159", "tiger_cat"]
losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    (LPNorm(model.input), 1),
    (TotalVariation(model.input), 1)
]
opt = Optimizer(model.input, losses)
max_iter = 100 #500
img, grads, wrt_value = opt.minimize( seed_input=seed_img, max_iter=max_iter, verbose=True, 
#callbacks=[GifGenerator('opt_progress')], # error in img = Image.fromarray(img)
#image_modifiers=[Jitter()],  # this option not available in the installed ver /gne/home/daij12/.local/lib/python3.6/site-packages/keras_vis-0.4.1-py3.6.egg/vis/
)

import numpy as np
imsave('test.png',np.hstack((seed_img, img)))
