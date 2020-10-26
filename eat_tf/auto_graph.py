import tensorflow as tf

'''
    tf2.0支持动态图，更加灵活，效率偏低。使用auto_graph将动态图的部分转化为静态图，
    提高执行效率，将函数使用@tf.function修饰，有三个限制
    1.函数内尽量使用tf的函数(tf.print， tf.constant(True))，其他函数无法嵌入静态图中
    2.不要定义tf.Variable
    3.不要改变外部python的列表或者字典等变量，无法嵌入静态图中修改（只读）
'''


# tf首先将函数内容转换为静态结构的python代码然后运行,得到定义好的图接着执行graph.run()
# 如果传入的不是tensor,每次都要执行上面的内容，最好传入tensor,不然没必要用auto_graph
@tf.function
def join_func(a, b):
    for i in tf.range(1, 3):
        tf.print(i)
    c = tf.strings.join((a, b), separator=' ')
    print("tracing")
    return c


x = tf.Variable(1.0,dtype=tf.float32)


# 使用tf.Variable可以将变量定义在函数外部
@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
def add_func(a):
    x.assign_add(a)
    return x


if __name__ == '__main__':
    c = join_func(tf.constant("hello"), tf.constant("world"))
    tf.print(c)
    d = join_func(tf.constant("good"), tf.constant("tf"))
    tf.print(d)
    tf.print(x)
    tf.print(add_func(tf.constant(1.0, tf.float32)))
