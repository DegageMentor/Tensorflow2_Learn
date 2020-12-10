import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt

# tf.random.set_seed(2)
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0;

    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
            if index >= len(imgs):
                break
        if index >= len(imgs):
            break
    new_im.save(name)


h_dim = 20
batch_size = 50

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
print(x_train.shape, x_test.shape)

# 无监督学习不需要label
db_train = tf.data.Dataset.from_tensor_slices(x_train)
db_train = db_train.shuffle(1000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)

sample = next(iter(db_test))
print(sample.shape)


class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(h_dim)
        ])

        # Decoder
        self.decoder = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None, mask=None):

        h = self.encoder(inputs)

        x_hat = self.decoder(h)

        return x_hat


def main():
    model = AE()
    model.build(input_shape=(None, 784))
    model.summary()

    optimizer = tf.optimizers.Adam(lr=1e-3)

    for epoch in range(10):

        for step, x in enumerate(db_train):

            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                logits = model(x)

                loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, logits, from_logits=True))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("epoch {}, step {}, loss = {}".format(epoch, step, loss))

        # evaluation
        x = next(iter(db_test))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'ae_images/re_epoch_%d step%d.jpg'% (epoch, step))


if __name__ == '__main__':
    main()
