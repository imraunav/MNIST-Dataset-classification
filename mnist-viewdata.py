from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
print(f'Train: feature = {train_features.shape}, labels = {train_labels.shape}')
print(f'Test: feature = {test_features.shape}, labels = {test_labels.shape}')
# plt.suptitle('Some of the training images')
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_features[i], cmap='gray')
    # plt.title(train_labels[i])
    plt.axis('off')

# plt.tight_layout()
plt.savefig('MNIST-Dataset-classification/mnist-datasample.png')
plt.show()
plt.close('all')