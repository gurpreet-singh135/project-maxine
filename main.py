# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import timeit
import tensorflow as tf
from pixgen import *
from patchdiscriminator import *
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Define a Python function
def function_to_get_faster(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Create an oveerride model to classify pictures
class SequentialModel(tf.keras.Model):
  def __init__(self, **kwargs):
    super(SequentialModel, self).__init__(**kwargs)
    self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
    # Add a lot of small layers
    num_layers = 100
    self.my_layers = [tf.keras.layers.Dense(64, activation="relu")
                      for n in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.dense_2 = tf.keras.layers.Dense(10)
  @tf.function
  def call(self, x):
    x = self.flatten(x)
    for layer in self.my_layers:
      x = layer(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    return x

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # Create a `Function` object that contains a graph
    # a_function_that_uses_a_graph = tf.function(function_to_get_faster)
    #
    # # Make some tensors
    # x1 = tf.constant([[1.0, 2.0]])
    # y1 = tf.constant([[2.0], [3.0]])
    # b1 = tf.constant(4.0)
    #
    # # It just works!
    # start = time.time()
    # function_to_get_faster(x1,y1,b1)
    # end = time.time()
    # print(end-start)
    # start = time.time()
    # a_function_that_uses_a_graph(x1, y1, b1).numpy()
    # end = time.time()
    # print(end-start)
    # model = SequentialModel()
    # input_data = tf.random.uniform([20, 28, 28])
    # model(input_data)
    # print("Graph time:", timeit.timeit(lambda: model(input_data), number=100))
    # model = PatchDiscriminator()
    # print(model(tf.random.normal([4,7,256,256,3])))
    # model.summary()
    # for layer in model.layers:
    #     print(layer.output_shape)
    model = PixGenerator(input_shape=[4,256,256,3])
    print(model(tf.random.normal([1,4,256,256,3])))
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/