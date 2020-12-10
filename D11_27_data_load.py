import tf_learning
import tensorflow as tf
from tensorflow import keras

# MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)    # (60000, 28, 28)
print(y_train.shape)    # (60000,)
print(x_test.shape)     # (10000, 28, 28)
print(y_test.shape)     # (10000,)
print(type(x_train))    # <class 'numpy.ndarray'>
print(type(y_train))    # <class 'numpy.ndarray'>

# y_test = [1., 2., 3., 4., 5., 6., 7., 8., 12., 10.]
# print(y_test[:10])
#
# y_test_onehot = tf.one_hot(y_test, depth=10)
# print(y_test_onehot[:10])


# CIFAR10
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# tf.data.Dataset
print("\n\n")
print(y_test[:4])
db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

print(db)
for i in range(4):
    print(next(iter(db))[0].shape)
    print(next(iter(db))[1])



