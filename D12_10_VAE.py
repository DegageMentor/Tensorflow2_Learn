import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt

tf.random.set_seed(2)
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0

    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
            if index == len(imgs):
                break;
        if index == len(imgs):
            break

    new_im.save(name)


z_dim = 10
batch_size = 512

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
print(x_train.shape, x_test.shape)

# 无监督学习不需要label
db_train = tf.data.Dataset.from_tensor_slices(x_train)
db_train = db_train.batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size)

sample = next(iter(db_test))
print(sample.shape)


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)  # get mean prediction
        self.fc3 = layers.Dense(z_dim)  # get variance prediction

        # Decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)   # 默认N(0, 1)
        std = tf.exp(log_var * 0.5)     # 计算标准差
        z = mu + std * eps

        return z

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encoder(inputs)

        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)

        return x_hat, mu, log_var


def main():
    model = VAE()
    model.build(input_shape=(4, 784))
    model.summary()

    optimizer = keras.optimizers.Adam(lr=1e-3)

    for epoch in range(10):

        for step, x in enumerate(db_train):
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                x_rec_logits, mu, log_var = model(x)

                # rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)    # shape = [512,]
                # rec_loss = tf.reduce_mean(rec_loss)   # Reduce_mean会导致rec_loss相比kl过小，重建效果不好，除非降低loss中kl比重，改为0.01

                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(x, x_rec_logits)     # shape[512, 784]
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

                # compute KL divergence (mu, var) ~ N(0, 1)
                kl_div = -0.5 * (log_var + 1 - mu**2 - tf.exp(log_var))
                # kl_div = tf.reduce_mean(kl_div)

                kl_div = tf.reduce_sum(kl_div) / x.shape[0]

                loss = rec_loss + 10.0 * kl_div

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("epoch {}, step {}, kl_div = {}, loss = {}".format(epoch, step, kl_div, loss))

        # evaluation
        # 随机sample
        x = tf.random.normal([90, z_dim])
        x_hat = model.decoder(x)
        x_hat_sig = tf.sigmoid(x_hat)

        x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
        x_hat_sig = tf.reshape(x_hat_sig, [-1, 28, 28]).numpy() * 255.

        save_images(x_hat.astype(np.uint8), 'vae_imgs/epoch{}.jpg'.format(epoch))
        save_images(x_hat_sig.astype(np.uint8), 'vae_imgs/epoch{}_sig.jpg'.format(epoch))

        # reconstruction
        x = next(iter(db_test))
        x = tf.reshape(x, [-1, 784])
        x_hat, _, _ = model(x)
        x_hat_sig = tf.sigmoid(x_hat)

        x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
        x_hat_sig = tf.reshape(x_hat_sig, [-1, 28, 28]).numpy() * 255.

        save_images(x_hat.astype(np.uint8), 'vae_imgs/rec_epoch{}.jpg'.format(epoch))
        save_images(x_hat_sig.astype(np.uint8), 'vae_imgs/rec_epoch{}_sig.jpg'.format(epoch))


if __name__ == '__main__':
    main()
