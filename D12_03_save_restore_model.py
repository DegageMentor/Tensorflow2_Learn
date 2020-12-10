import tf_learning

import tensorflow as tf
from tensorflow import keras


def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_val, y_val) = keras.datasets.mnist.load_data()
print(x.shape, y.shape)

batch_size = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(100).batch(batch_size)  # 处理顺序很重要

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(batch_size)

network = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10)])

network.build(input_shape=(None, 28*28))
network.summary()

network.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics='accuracy')

# network.fit(db, epochs=10, validation_data=db_test, validation_freq=1)

network.fit(db, epochs=5, validation_data=db_val, validation_freq=1)

# # 方法1  sava_weights法
# network.save_weights('./model/m2.ckpt')
# del network
# # 加载权重
# model = keras.Sequential([
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(10)])
#
# model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
#                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics='accuracy')
#
# model.load_weights('./model/m2.ckpt')

# 方法2 save_model法（效率较低）
network.save('./model/m4.h5')
del network
#加载模型
print('\nload model')

# 直接加载整个模型
model = tf.keras.models.load_model('./model/m4.h5')

model.evaluate(db_val)

sample = next(iter(db_val))
x = sample[0]
y = sample[1]  # one-hot
pred = model.predict(x)  # [b, 10]
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)


# # 方法3   该方法 不能直接调用evaluate， predict...
# tf.saved_model.save(network, 'model/fangfa3/m5.h5')
# del network
# model = tf.saved_model.load('model/fangfa3/m5.h5')
# f = model.signatures["serving_default"]
#
# print(type(f))




