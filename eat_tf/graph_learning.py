import tensorflow as tf
from datetime import datetime
import os


def join_str(x, y, sep=' '):
    result = tf.strings.join([x, y], separator=sep)
    return result


@tf.function
def join_str_auto(x,y):
    z = tf.strings.join([x, y],separator=" ")
    return z


if __name__ == '__main__':
    x = tf.constant("hello")
    y = tf.constant("world")

    # 动态图
    print(join_str(x, y))

    # autograph
    print(join_str_auto(x, y))

    # 求梯度
    x = tf.Variable(0.0, name="x", dtype=tf.float32)
    a = tf.Variable(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c
    print(tape.gradient(y, x))

    # 二阶导数
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape2.gradient(y, x)
    dy_dx2 = tape1.gradient(dy_dx, x)
    print(dy_dx2)

    # 利用SGD求最小值
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for _ in range(1000):
        with tf.GradientTape() as tape:
            y = a * x ** 2 + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
    tf.print("y =", y, "; x =", x)
