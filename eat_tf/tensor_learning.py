import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    # int8, int16, int32, int64
    x_1 = tf.constant(1)
    x_2 = tf.constant(1, dtype=tf.int64)
    # tf.float16 float32 float64
    x_3 = tf.constant(1.33)
    x_4 = tf.constant(3.14, tf.float64)
    x_5 = tf.constant("temp_str")
    x_6 = tf.constant(True, dtype=tf.bool)

    # 和numpy的类型对比
    print(tf.int64 == np.int64)
    print(tf.float64 == np.float64)
    print(tf.bool == np.bool)
    print(tf.string == np.str_)

    # 计算维度
    scalar_x = tf.constant(1)
    print(tf.rank(scalar_x))
    print(scalar_x.numpy().ndim)

    vector_x = tf.constant([1, 2, 3, 4, 5])
    print(tf.rank(vector_x))
    print(vector_x.numpy().ndim)

    matrix_x = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(tf.rank(matrix_x))
    print(matrix_x.numpy().ndim)

    # cast函数
    scalar_y = tf.cast(scalar_x, tf.float64)
    print(scalar_y)

    # shape函数
    print(vector_x.shape)

    # 变量
    var = tf.Variable([1.0, 2.0])
    var.assign_add([1.0, 3.1])
    print(var)

