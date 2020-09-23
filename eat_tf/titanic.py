import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 对数据进行预处理
def pre_process(dataset):

    result_ds = pd.DataFrame()

    # one-hot label
    class_ds = pd.get_dummies(dataset['Pclass'])
    class_ds.columns = ["Pclass_" + str(x) for x in class_ds.columns]
    result_ds = pd.concat([result_ds, class_ds], axis=1)

    # sex
    sex_ds = pd.get_dummies(dataset['Sex'])
    result_ds = pd.concat([result_ds, sex_ds], axis=1)

    # age填充0，增加一维度特征标识是否年龄为空
    result_ds['Age'] = dataset['Age'].fillna(0)
    result_ds['Age_null'] = pd.isna(dataset['Age']).astype('int32')

    # SibSp,Parch,Fare三维数值型特征，没有缺失值
    result_ds['SibSp'] = dataset['SibSp']
    result_ds['Parch'] = dataset['Parch']
    result_ds['Fare'] = dataset['Fare']

    # cabin特征保留是否为null
    result_ds['Cabin_null'] = pd.isna(dataset['Cabin']).astype('int32')

    # one-hot
    embarked_ds = pd.get_dummies(dataset['Embarked'], dummy_na=True)
    embarked_ds.columns = ['Embarked_' + str(x) for x in embarked_ds.columns]
    result_ds = pd.concat([result_ds, embarked_ds], axis=1)

    # label
    result_ds['label'] = dataset['Survived']

    return result_ds


# 构造模型
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(20, activation='relu', input_shape=(15, )))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


# 画出每个epoch的训练集和测试集的loss
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()


if __name__ == '__main__':
    dataset = pre_process(pd.read_csv('../resources/titanic/train.csv'))
    print(dataset.dtypes)
    features = dataset.iloc[:, 0:-1]
    labels = dataset['label'].values
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=0)

    simple_model = get_model()
    simple_model.compile(optimizer='adam',
                         loss=tf.keras.losses.binary_crossentropy,
                         metrics=['AUC'])

    history = simple_model.fit(train_features, train_labels,
                               batch_size=64,
                               epochs=30,
                               validation_split=0.2
                               )

    # plot_metric(history, "loss")
    # plot_metric(history, "AUC")

    # 保存权重、模型
    simple_model.save_weights('../resources/titanic/ckp/simple_model.ckp', save_format='tf')
    simple_model.save('../resources/titanic/simple_model/', save_format='tf')

    # 加载模型
    model_loaded = tf.keras.models.load_model("../resources/titanic/simple_model/")

    # 预测
    predicts = model_loaded.evaluate(x=test_features, y=test_labels)
    print(predicts)



