import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def decode_review(text, index2word):
    return ' '.join([index2word.get(i, '?') for i in text])

if __name__ == '__main__':

    vocab_size = 10000

    imdb = keras.datasets.imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

    print(train_data[0])
    print(len(train_data[0]), len(train_data[1]))

    # (word->index),index从1开始
    word2index = imdb.get_word_index()

    # 增加4个索引
    word2index = {k: (v + 3) for k, v in word2index.items()}
    word2index["<PAD>"] = 0
    word2index["<START>"] = 1
    word2index["<UNK>"] = 2  # unknown
    word2index["<UNUSED>"] = 3

    # 转换为(index->word)
    index2word = dict([(value, key) for key, value in word2index.items()])

    # 把数据转换为文本
    for i in range(100):
        print(decode_review(train_data[i], index2word))

    # padding
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word2index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word2index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    print(decode_review(train_data[0], index2word))

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    history = model.fit(x_train,
                        y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(test_data, test_labels, verbose=2)

    for k, v in zip(model.metrics_names, results):
        print('name : %s, value : %f' % (k, v))

    # 作图
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # 清除数字
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

