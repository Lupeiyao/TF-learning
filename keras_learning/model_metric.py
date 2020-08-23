import tensorflow as tf
import numpy as np


class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name="categorical_true_positives_metric", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, tf.int32), tf.cast(y_true, tf.int32))

        y_true = tf.cast(y_true, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)

        return self.true_positives.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return tf.identity(self.true_positives)

    def reset_states(self):
        self.true_positives.assign(0.)


if __name__ == '__main__':

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[CategoricalTruePositives()])

    train_x = np.random.random((10000, 10))
    train_y = np.random.randint(0, 10, (10000, 1))

    model.fit(train_x, train_y, epochs=10, batch_size=32, validation_split=0.2)

    model.summary()

