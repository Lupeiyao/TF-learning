import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class SimpleDeepWide(tf.keras.Model):

    def __init__(self, config):
        super(SimpleDeepWide, self).__init__(name="deep_wide")
        self._mlp = [tf.keras.layers.Dense(config['units_list'][i], activation='relu', name="deep_{}".format(i)) for i in range(len(config['units_list']))]
        self._output = tf.keras.layers.Dense(1, activation="sigmoid", name="out")

    def call(self, inputs):
        deep_out = inputs
        for layer in self._mlp:
            deep_out = layer(deep_out)
        wide_out = inputs
        result = self._output(tf.keras.layers.concatenate([deep_out, wide_out]))
        return result


if __name__ == '__main__':

    model = SimpleDeepWide({"units_list": [32, 32, 32, 32]})
    train_x = np.random.random((10000, 100))
    train_y = np.random.randint(0, 2, (10000, 1))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['AUC'])
    model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.2)
    model.summary()






