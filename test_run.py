


from test import *
import time




if __name__ == '__main__':
    print('hello')
    layer = MyDenseLayer(10)
    # print(layer(tf.random.normal([4,7,28,28,1])))
    # layer.build(input_shape=(None,10,5))
    # layer.summary()
    start = time.time()
    print(layer(tf.zeros([4, 10, 5])))
    end = time.time()
    print(end-start)
    start = time.time()
    print(layer(tf.zeros([4, 10, 5])))
    end = time.time()
    print(end-start)

    layer = MyConv2Dtranspose(10, (56,56),3,2,'SAME',tf.random_normal_initializer(0., 0.02))
    start = time.time()
    print(layer(tf.zeros((4,7,28,28,1))).shape)
    end = time.time()
    print(end-start)
    start = time.time()
    print(layer(tf.zeros((4,7,28,28,1))).shape)
    end = time.time()
    print(end-start)