# DEEP LEARNING USING KERAS TUTORIAL

## INTRODUCTION
This post will take you through a simple implementation of **convolutional neural netwotks** using [keras](https://keras.io/) for classification of **MNIST** dataset.

Keras is a deep learning library built over [theano](deeplearning.net/software/theano/) and [tensorflow](https://www.tensorflow.org/).It is very easy for beginners to get started on neural networks implementation using keras.So let's start by importing the essentials.You can learn to install them from [here](installation)
```python
import numpy as np
import cv2
from matplotlib import pyplot as plt 
```
NumPy is the fundamental package for scientific computing with Python.OpenCV is used along with matplotlib just for showing some of the results in the end.
```python
from keras.models import Sequential 
```
The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model.It is a linear stack of neural network layers for feed forward cnn
```python
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D
```
Now we are importing core layers for our CNN netwrok.Dense layer is actually a [fully-connected layer](http://cs231n.github.io/convolutional-networks/#fc).[Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) is a regularization technique used for reducing overfitting by randomly dropping output units in the network.Activation Function is used for introducing
non-linearity and Flatten is used for converting output matrices into a vector.

The second line is used for importing convolutional and pooling layers.
```python
from keras.utils import np_utils
```
This library will be useful for converting shape of our data.
```python
from keras.datasets import mnist
```
MNIST dataset is already available in keras library.So, you dont need to download it separately.
