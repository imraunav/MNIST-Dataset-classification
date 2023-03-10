from matplotlib import pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfds # will learn to use this in the future
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path='/Users/raunavghosh/Documents/Research for Vehant/MNIST/MNIST-Dataset-classification/mnist.npz')

print('MNIST Dataset Shape:')
print(f'training images: {train_images.shape}') # (60000, 28, 28)
print(f'training labels: {train_labels.shape}') # 60000,)
print(f'testing images: {test_images.shape}') # (10000, 28, 28)
print(f'testing labels: {test_labels.shape}') # (10000,)
# plt.suptitle('Some of the training images')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i], cmap='gray')
    # plt.title(train_labels[i])
    plt.axis('off')

# plt.tight_layout()
plt.savefig('MNIST-Dataset-classification/mnist-datasample.png')
plt.show()
plt.close('all')