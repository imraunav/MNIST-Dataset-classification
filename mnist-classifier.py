from matplotlib import pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfds # will learn to use this in the future
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path='/Users/raunavghosh/Documents/Research for Vehant/MNIST/MNIST-Dataset-classification/mnist.npz')
