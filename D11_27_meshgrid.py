import tf_learning
import tensorflow as tf
from matplotlib import pyplot as plt

def fuc(x):
    z = tf.math.sin(x[..., 0]) + tf.math.sin(x[..., 1])
    return z

x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)

point_x, point_y = tf.meshgrid(x, y)
print(point_x.shape)

points = tf.stack([point_x, point_y], axis=2)
# points = tf.reshape(points, [-1, 2])
print(points.shape)

z = fuc(points)
print(z.shape)

plt.figure('2d func value')
plt.imshow(z, origin='lower')
plt.colorbar()

plt.figure('2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()

