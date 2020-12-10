import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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

        # 初始化0状态
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]

        # transform text to embedding
        # [b, 80] => [b, 80, 100]
        self.embedding = keras.layers.Embedding(total_words, embedding_len, input_length=sentence_len)

        # [b, 80, 100] => [b, units]
        self.rnn_cell0 = keras.layers.SimpleRNNCell(units, dropout=0.2)

        self.rnn_cell1 = keras.layers.SimpleRNNCell(units, dropout=0.2)

        # [b, units] => [b, 1]
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):

        x = self.embedding(inputs)

        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)  # 有dropout处理要传入training参数
            out1, state1 = self.rnn_cell1(out0, state1, training)

        x = self.fc(out1)

        prob = tf.nn.sigmoid(x)

        return prob


def main():

    units = 64
    epochs = 5

    model = MyRNN(units)
    model.build(input_shape=(None, 80))     # input_shape必须是个turtle
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics='acc')

    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


    # # test
    # total_num, total_correct = 0, 0
    # for x, y in db_test:
    #     logits = model(x)
    #
    #     # prob = tf.nn.softmax(logits, axis=1)
    #     preds = list(map(lambda x: 1 if x > 0.5 else 0, logits.numpy()))
    #     preds = tf.cast(preds, dtype=tf.int32)
    #     y = tf.cast(y, dtype=tf.int32)
    #
    #     correct = tf.reduce_sum(tf.cast(tf.equal(preds, y), dtype=tf.int32))
    #
    #     total_correct += correct
    #     total_num += x.shape[0]
    #
    # acc = total_correct / total_num
    # print("test acc = ", acc.numpy(), "total_num1 = ", total_num)


if __name__ == '__main__':
    main()
