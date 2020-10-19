import tensorflow as tf
import time


if __name__ == '__main__':
    # 常规tensor创建
    vector_x = tf.constant([1, 2, 3], tf.int32)
    range_x = tf.range(1, 10, delta=2)
    linSpace_x = tf.linspace(0.0, 2 * 3.14, 100)
    zeros_x = tf.zeros([3, 3])
    ones_x = tf.ones([3, 3])
    fill_x = tf.fill([3, 3], 5)

    # 分布类型创建
    tf.random.set_seed(time.time())

    # 随机均匀分布、正太分布、剔除2倍方差以外的正太分布
    rand_x = tf.random.uniform([5], minval=0, maxval=10)
    normal_x = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
    truncated_normal_x = tf.random.truncated_normal((5, 5), mean=0.0, stddev=1.0)

    # 单位矩阵、对角矩阵
    eye_x = tf.eye(3, 3)
    diag_x = tf.linalg.diag([1, 2, 3])

    # 切片操作
    uniform_matrix = tf.random.uniform([5, 5, 5], minval=0, maxval=10, dtype=tf.int32)
    tf.print(uniform_matrix)
    tf.print(uniform_matrix[0][1][3])
    tf.print(uniform_matrix[1:4, 0:3, ::2])
    tf.print(tf.slice(uniform_matrix, [1, 0, 0], [3, 3, 2]))

    var_x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
    var_x[1, :].assign(tf.constant([0.0, 0.1]))
    tf.print(var_x)
