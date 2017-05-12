## Key Ingredients for Deep Learning 


- Hardware at the core: Nvidia GPU
  - Kepler eg K80
  - Maxwell eg M60
  - Pascal ([wiki](https://en.wikipedia.org/wiki/Pascal_(microarchitecture))) eg P100
  - Volta (to come)

- System

- Computing model: [CUDA](https://developer.nvidia.com/cuda-downloads) ([wiki](https://en.wikipedia.org/wiki/CUDA))
  - CUDA 7
  - CUDA 8

- Deep Learning frameworks (name, backed-by)

  - tensorflow, Google
  - keras, Google
  - caffe/caffe2, Facebook
  - torch/pytorch, Facebook
  - mxnet, Amazon for AWS
  - cntk, Microsoft
  - Theano,
  - [dl4j](https://deeplearning4j.org/),


- Model architecture

  - LeNet
  - AlexNet
  - VGGNet
  - GoogLeNet
  - Inception
  - Xception
  - ResNet

- Special purposed architecture

  - U-net ([arxiv](https://arxiv.org/abs/1505.04597))
  - V-net
  - E-net

- Building blocks
  - densely connected layer ([my tutorial](https://github.com/jiandai/mlTst/blob/master/tensorflow/ann101.ipynb))
  - locally connected layer
  - convolutional layer ([my tutorial](https://github.com/jiandai/mlTst/blob/master/semeion.ipynb))
  - pooling layer
  - dropout layer

- Techniques
  - Optimization (a lot)  
  - Batch normalization
  - Data augmentation ([my tutorial](https://github.com/jiandai/mlTst/blob/master/keras/image-data-augmentation-by-keras.ipynb))
  - Feature extraction using pretained weight ([my tutorial](https://github.com/jiandai/mlTst/blob/master/keras/DL-features.ipynb))
  - Transfer learning ([my tutorial using dogs-vs-cats data](https://github.com/jiandai/dogs-vs-cats/blob/master/dogs-vs-cats-ex.ipynb))
  - Visualization computational graph and training process ([my tutorial (under construction)](https://github.com/jiandai/mlTst/blob/master/tensorflow/tensorboard-101.ipynb))
