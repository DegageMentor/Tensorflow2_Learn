import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


# 卷积层网络结构 conv + maxpooling VGG13
conv_layers = [
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.Conv2D(128, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]


# 数据预处理函数
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 加载CIFATR100数据集

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test)
print("shape: ", x.shape, y.shape, x_test.shape, y_test.shape)

batch_size = 200
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batch_size)


sample = next(iter(train_db))
print(type(sample))
print(tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():

    conv_net = Sequential(conv_layers)
    conv_net.build(input_shape=[None, 32, 32, 3])

    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    # print(out.shape)

    # 定义全连接层
    fc_net = Sequential([layers.Dense(256, activation='relu'),
                         layers.Dense(128, activation=tf.nn.relu),
                         layers.Dense(100, activation=None)])

    fc_net.build(input_shape=[None, 512])

    # [1, 2] + [3, 4] = [1, 2, 3, 4] list相加
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    print("variales num = ", len(variables))
    # for v in variables:
    #     print(v.shape)

    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(10):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # flatten
                out = tf.reshape(out, [-1, 512])
                # [b, 512] => [b, 100]
                logits = fc_net(out)

                y_onehot = tf.one_hot(y, depth=100)

                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print("epoch = ", epoch, "step = ", step, "loss = ", loss.numpy())

        # test
        total_num1, total_num2, total_correct = 0, 0, 0
        for x, y in test_db:
            out = conv_net(x)
            logits = fc_net(tf.reshape(out, [-1, 512]))
            # prob = tf.nn.softmax(logits, axis=1)
            preds = tf.argmax(logits, axis=1)
            preds = tf.cast(preds, dtype=tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(preds, y), dtype=tf.int32))

            total_correct += correct
            total_num1 += batch_size
            total_num2 += x.shape[0]

        acc = total_correct / total_num2
        print("test acc = ", acc.numpy(), "total_num1 = ", total_num1, "total_num2 = ", total_num2)


if __name__ == '__main__':
    main()