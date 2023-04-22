import tensorflow as tf
from tensorflow.keras import layers, Model, Input, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import numpy as np

def make_generator():
    noise = Input(shape=(1,))
    h0 = layers.Dense(units=1200, activation='relu', name="h0")(noise)
    h1 = layers.Dense(units=1200, activation='relu', name="h1")(h1)
    y = layers.Dense(units=784, activation='sigmoid', name="y")(h1)
    gen_img = layers.Reshape((28,28,1),name="Gen-img")
    return Model(inputs=[noise,],
                 outputs=[gen_img,],
                 name="Generator")

def make_discriminator():
    # don't have maxout activation fucntions so using relu instead
    img = Input(shape=(28,28,1))
    h0 = layers.MaxoutDense(units=240, activation='relu', name="h0")(img)
    h1 = layers.MaxoutDense(units=240, activation='relu', name="h1")(h0)
    y = layers.Dense(units=1, activation='sigmoid', name="y")(h1)
    return Model(inputs=[img,],
                 outputs=[y,],
                 name="Discriminator")

def myGan(Model):
    def __init__(self):
        super().__init__()
        self.

