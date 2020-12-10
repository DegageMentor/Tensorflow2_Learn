import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    # [0~255] => [-1~1]
    x = 2. * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batch_size = 128

(x, y), (x_val, y_val) = datasets.cifar10.load_data()
print(y.shape, y_val.shape)     # [50k, 1], [10k, 1]

y = tf.squeeze(y)   # squeeze去掉大小为1的维度
y_val = tf.squeeze(y_val)
print(y.shape, y_val.shape)

y = tf.one_hot(y, depth=10)             # [50k, 10]
y_val = tf.one_hot(y_val, depth=10)     # [10k, 10]
print(y.shape, y_val.shape)

print("datasets:", x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(10000)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)


# 自定义Densse层，移除bias
class MyDense(layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight('w', [inp_dim, outp_dim])
        # self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel

        return x


# 自定义Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        '''

        :param inputs: [b, 32, 32, 3]
        :param training:
        :param mask:
        :return:
        '''
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])

        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)

        return x


# 调用自定义模型
model = MyModel()

model.compile(optimizer=optimizers.Adam(),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_db, epochs=10, validation_data=test_db, validation_freq=1)


# 保存模型
model.save_weights('model/cifar10/')


# 加载模型

new_model = MyModel()
new_model.compile(loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics='acc')
new_model.load_weights('model/cifar10/')

new_model.evaluate(test_db)
