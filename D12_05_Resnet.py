import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


# basic block
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, [3, 3], strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, [3, 3], strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = layers.Conv2D(filter_num, [1, 1], strides=stride, padding='same')
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        # inputs [b, h, w, c]

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = self.relu(output)

        return output


# ResNet
class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100): # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, [3, 3], strides=1, padding='same'),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
                                ])

        self.layer1 = self.build_resblock(64, layer_dims[0])

        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)

        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)

        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgbool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        # [b, c]
        x = self.avgbool(x)

        # [b, 100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()

        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])


def main():
    x = tf.random.normal([4, 32, 32, 3])
    model = resnet18()

    print(model(x).shape)


if __name__ == '__main__':
    main()