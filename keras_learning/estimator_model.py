from tensorflow import keras
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


def input_fn():
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__(name="simple_model")
        self._mlp = [keras.layers.Dense(32, activation='relu', name="mlp_{}".format(i))
                     for i in range(5)]
        self._out = keras.layers.Dense(3, activation='softmax', name="out")

    def call(self, inputs, **kwargs):
        output = inputs
        for layer in self._mlp:
            output = layer(output)
        output = self._out(output)
        return output


if __name__ == '__main__':
    # model = MyModel()

    # 目前只支持这种堆叠的方式构造estimator
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    import tempfile

    model_dir = tempfile.mkdtemp()
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir)

    keras_estimator.train(input_fn=input_fn, steps=500)
    eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
    print('Eval result: {}'.format(eval_result))