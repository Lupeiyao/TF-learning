import tensorflow as tf
import numpy as np
from sklearn import datasets
import pandas as pd


def parse_tfrecord(pb):
    description = {
        "label" : tf.io.FixedLenFeature([], tf.int32),
        "feature" : tf.io.FixedLenFeature([], tf.float64)
    }
    example = tf.io.parse_single_example(pb, description)
    return example['feature'], example['label']


if __name__ == '__main__':
    # 从numpy数组构造dataset
    iris = datasets.load_iris()
    data_set1 = tf.data.Dataset.from_tensor_slices((iris['data'], iris['target']))
    for feature, label in data_set1.take(3):
        # 两个tensor
        print(feature, label)

    # 从pandas的dataframe构造dataset，map+
    df_iris = pd.DataFrame(iris["data"], columns=iris.feature_names)
    data_set2 = tf.data.Dataset.from_tensor_slices((df_iris.to_dict('list'), iris['target']))
    for feature, label in data_set2.take(3):
        # 一个map,一个tensor
        print(feature, label)

    # dataset支持map、filter、flat_map、reduce、
    # batch、padded_batch、repeat、zip、concatenate（拼接）
    # window（返回dataset[dataset]）
    file_list = tf.data.Dataset.list_files("../resources/titanic/*.csv")

    # 使用interleave可以高效并行
    # 1.获取cycle_length个数据执行function，
    # 2.从每个function返回的dataset中依次获取block_length个数据直到每个dataset都取完
    # 3.再获取cycle_length个数据，循环
    file_list.interleave(lambda file_path: tf.data.TFRecordDataset(file_path),
                         cycle_length=10,
                         block_length=32)\

    # map可以设置并行度提高效率，还可以先batch再对batch后的数据进行map
    data_set3 = file_list.map(lambda file_path: tf.data.TFRecordDataset(file_path),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .flat_map(lambda pb: parse_tfrecord(pb))\
        .batch(32)\
        .repeat(10)\
        .prefetch(1)