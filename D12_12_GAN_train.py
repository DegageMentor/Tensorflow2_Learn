import tf_learning
import  os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from PIL import Image
import  glob
from    D12_12_GAN import Generator, Discriminator

from    D12_12_dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, training):
    fake_img = generator(batch_z, training)

    d_fake_logits = discriminator(fake_img, training)
    d_real_logits = discriminator(batch_x, training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    return d_loss_fake + d_loss_real


def g_loss_fn(generator, discriminator, batch_z, training):
    fake_img = generator(batch_z, training)

    d_fake_logits = discriminator(fake_img, training)

    g_loss = celoss_ones(d_fake_logits)

    return g_loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)

    # 超参数
    z_dim = 100
    epochs = 30000
    batch_size = 64
    learing_rate = 0.002
    training = True

    imgs = glob.glob(r'H:\faces\faces\*.jpg')
    print("images num = ", len(imgs))
    dataset, img_shape, _ = make_anime_dataset(imgs, batch_size)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    diter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(4, z_dim))
    generator.summary()
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    discriminator.summary()

    g_optimizer = tf.optimizers.Adam(learning_rate=learing_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learing_rate, beta_1=0.5)

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], maxval=1., minval=-1.)
        batch_x = next(diter)

        # train Discriminator
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train Generator
        # batch_z = tf.random.uniform([batch_size, z_dim], maxval=1., minval=-1.)
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print("epoch {}, d_loss = {}, g_loss = {}".format(epoch, d_loss, g_loss))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join(r'H:\faces\gan', 'epoch{}.jpg'.format(epoch))
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()