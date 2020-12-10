import tf_learning

import tensorflow as tf

x = tf.cast([[1, 2], [3, 4]], dtype=tf.float32)
w = tf.cast(tf.ones([2, 2]), dtype=tf.float32)
b = tf.zeros([2])

y = tf.constant([1, 0])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = x @ w + b
    # reduce_mean对每个样本的损失（样本与标签每个对应元素的平方和 / 标签元素个数）求平均
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=2), prob))

grads = tape.gradient(loss, [w, b])

print("loss = ", loss)
print('\n梯度:')
print(grads[0])
print(grads[1])