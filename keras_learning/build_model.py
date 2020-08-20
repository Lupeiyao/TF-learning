import tensorflow as tf
import numpy as np


class DNN(tf.keras.Model):

    def __init__(self, params):
        super(DNN, self).__init__(name="dnn")
        self._mlp_layers = [tf.keras.layers.Dense(params['layers'][i],
                                                  activation='relu') for i in range(len(params['layers']))]
        self._out_layer = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs):
        output = inputs
        for i in range(len(self._mlp_layers)):
            output = self._mlp_layers[i](output)
        output = self._out_layer(output)
        return output


if __name__ == '__main__':

    train_x = np.random.random((10000, 72))
    # train_y = np.random.randint(0, 3, (10000, 1))
    train_y = np.random.random((10000, 3))
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).repeat()

    test_x = np.random.random((2000, 72))
    # test_y = np.random.randint(0, 3, (2000, 1))
    test_y = np.random.random((2000, 3))

    config = {"layers": [32, 32, 32]}
    model = DNN(config)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    history = model.fit(ds, epochs=10, steps_per_epoch=313)
    model.evaluate(test_x, test_y, batch_size=10)

    model.summary()
