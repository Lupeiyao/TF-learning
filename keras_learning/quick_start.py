import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    # 搭建模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['AUC'])

    # 构造训练数据
    train_data = tf.data.Dataset.from_tensor_slices((np.random.random((10000, 10)), np.random.randint(0, 3, (10000, 1))))
    train_data = train_data.batch(32).repeat()
    train_data.repeat()

    # 构造验证数据
    val_data = tf.data.Dataset.from_tensor_slices((np.random.random((500, 10)), np.random.randint(0, 3, (500, 1))))
    val_data = val_data.batch(32).repeat()

    # 构造测试数据
    test_data = tf.data.Dataset.from_tensor_slices((np.random.random((2000, 10)), np.random.randint(0, 3, (2000, 1))))
    test_data = test_data.batch(32).repeat()

    # 训练
    model.fit(train_data, epochs=10, steps_per_epoch=30,
              validation_data=val_data, validation_steps=3)

    # 评估
    model.evaluate(test_data, steps=20)

    # 预测
    result = model.predict(np.random.random((20, 10)), batch_size=2)

    print(result)
