import tf_learning

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 该函数有四个局部最优解，均=0，也是全局最优解   (3.0, 2.0)  (-2.8, 3.1)  (-3.7, -3.2)  (3.5, -1.8)
def himmelblau(x):
    return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print("X, Y range : ", x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X, Y maps : ', X.shape, Y.shape)
Z = himmelblau([X, Y])
print(Z.shape)

fig = plt.figure("himmelblau")
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# (0., 0.)
x = tf.constant([-4., 0.])

for step in range(200):

    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads

    if step % 20 == 0:
        print('step : {}, x = {}, y = {}'.format(step, x.numpy(), y.numpy()))
