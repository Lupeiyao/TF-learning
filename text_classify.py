import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

if __name__ == "__main__":

    # 下载数据
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)

    # 查看数据
    # train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    # print(train_examples_batch)
    # print(train_labels_batch)

    # 迁移学习（预训练的的embedding网络）
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)

    # print(hub_layer(train_examples_batch[:3]))

    # 构建模型（embedding + fc + sigmod）
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1))

    # 打印模型统计结构
    model.summary()

    model.compile(optimizer="adam",
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # 训练 + 验证
    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=20,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    # 测试
    results = model.evaluate(test_data.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

