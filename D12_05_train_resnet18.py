import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from D12_05_Resnet import resnet18


# 数据预处理函数
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=100)
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

    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    optimizer = optimizers.Adam(lr=1e-4)

    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics='acc')
    model.fit(train_db, epochs=10, validation_data=test_db, validation_freq=1)

    model.evaluate(test_db)

    # 训练10个epoch
    # for epoch in range(10):
    #
    #     for step, (x, y) in enumerate(train_db):
    #
    #         with tf.GradientTape() as tape:
    #             # [b, 32, 32, 3] => [b, 100]
    #             logits = model(x)
    #
    #             y_onehot = tf.one_hot(y, depth=100)
    #
    #             loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
    #             loss = tf.reduce_mean(loss)
    #
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #
    #         if step % 100 == 0:
    #             print("epoch = ", epoch, "step = ", step, "loss = ", loss.numpy())
    #
    #     # test
    #     total_num1, total_num2, total_correct = 0, 0, 0
    #     for x, y in test_db:
    #         logits = model(x)
    #
    #         # prob = tf.nn.softmax(logits, axis=1)
    #         preds = tf.argmax(logits, axis=1)
    #         preds = tf.cast(preds, dtype=tf.int32)
    #
    #         correct = tf.reduce_sum(tf.cast(tf.equal(preds, y), dtype=tf.int32))
    #
    #         total_correct += correct
    #         total_num1 += batch_size
    #         total_num2 += x.shape[0]
    #
    #     acc = total_correct / total_num2
    #     print("test acc = ", acc.numpy(), "total_num1 = ", total_num1, "total_num2 = ", total_num2)


if __name__ == '__main__':
    main()