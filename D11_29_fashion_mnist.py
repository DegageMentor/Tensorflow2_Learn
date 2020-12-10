import tf_learning

import tensorflow as tf
from tensorflow import keras


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# fashion_mnist 和mnist 数据集大小格式一样，图片内容不同
(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(x.shape, y.shape, x_test.shape, y_test.shape)
print(type(x), type(y))

batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
db = db.map(preprocess).shuffle(100)
db_iter = iter(db)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batch_size)

sample = next(db_iter)
print("\nbatch : ", sample[0].shape, sample[1].shape, type(sample), type(sample[0]), type(sample[1]))
sample_img = sample[0][0]
print(sample_img.shape)

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10)
    # keras.layers.Dense(10, activation='softmax')
])

model.build(input_shape=[None, 28 * 28])
model.summary()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

def main():

    for epoch in range(10):

        for step, (x, y) in enumerate(db):
            x = tf.reshape(x, [-1, 28 * 28])
            y_onehot = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_cross_entropy = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss_cross_entropy, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("epoch : ", epoch, "Step : ", step, "loss_mes = ", loss_mse.numpy(), "loss_ce = ", loss_cross_entropy.numpy())

        # test
        total_correct, total_num = 0, 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))

            total_correct += correct.numpy()
            total_num += x.shape[0]

        acc = total_correct / total_num
        print('test acc = ', acc, '\n')


if __name__ == '__main__':
    # main()
    print("..............")