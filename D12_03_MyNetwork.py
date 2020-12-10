import tf_learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 继承自keras.layers.Layer
class MyDense(layers.Layer):
    # 实现以下两个方法
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_weight('w', [inp_dim, outp_dim])   # 要使用add_variable方法添加变量(add_weight)
        self.bias = self.add_weight('b', [outp_dim])

    def call(self, inputs, training=None):

        out = inputs @ self.kernel + self.bias

        return out


# 继承自keras.Model
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)

        return x


# 数据预处理
def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


# 加载数据
(x, y), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
print(x.shape, y.shape)

batch_size = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(100).batch(batch_size)  # 处理顺序很重要

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batch_size)

# 使用自定义模型进行训练
network = MyModel()

network.compile(optimizer=keras.optimizers.Adam(),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics='accuracy')

network.fit(db, epochs=5, validation_data=db_val, validation_freq=1)

network.evaluate(db_val)