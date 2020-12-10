import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

print(tf.__version__)

batch_size = 128
total_words = 10000
sentence_len = 80
embedding_len = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=sentence_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=sentence_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)

print(tf.reduce_max(x_train), tf.reduce_max(x_test))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()

        # transform text to embedding
        # [b, 80] => [b, 80, 100]
        self.embedding = keras.layers.Embedding(total_words, embedding_len, input_length=sentence_len)

        # [b, 80, 100] => [b, units]
        self.rnn = keras.Sequential([
            # SimpleRNN
            # layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            # layers.SimpleRNN(units, dropout=0.5, unroll=True)

            # LSTM
            # layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=False),        # unrroll False GPU模式下可能更快
            # layers.LSTM(units, dropout=0.5, unroll=False)

            # GRU
            layers.GRU(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.GRU(units, dropout=0.5, unroll=True)

        ])

        # [b, units] => [b, 1]
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)

        x = self.rnn(x)
        x = self.fc(x)
        prob = tf.nn.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 5

    t0 = time.time()

    model = MyRNN(units)
    model.build(input_shape=(None, 80))  # input_shape必须是个turtle
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics='acc')

    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)

    t1 = time.time()
    print("total_time : ", t1 - t0)


if __name__ == '__main__':
    main()
