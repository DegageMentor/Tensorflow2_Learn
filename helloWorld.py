import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

print("Hello World")

print(tf.test.is_gpu_available())
