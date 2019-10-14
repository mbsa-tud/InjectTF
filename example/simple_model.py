#
#   TU-Dresden, Institute of Automation (IfA)
#
#   simple_model.py
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

# import modules
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# - - - - - - - - - - - - - - - - - - - -
# Setup:
# - - - - - - - - - - - - - - - - - - - -

model_dir = "./model/"
model_name = "simple_model"

# Those are the names of the input and output tensors of the network.
mappings = {"inputs": ["input_x"], "outputs": ["output_y", "output_y_"]}

# Training epochs:
TRAINING_EPOCHS = 5

BATCH_SIZE = 100

# Load the mnist dataset.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# - - - - - - - - - - - - - - - - - - - -
# Create the model:
# - - - - - - - - - - - - - - - - - - - -

x = tf.placeholder(tf.float32, [None, 784], name=mappings["inputs"][0])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b, name=mappings["outputs"][0])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10], name=mappings["outputs"][1])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


# - - - - - - - - - - - - - - - - - - - -
# Utility functions:
# - - - - - - - - - - - - - - - - - - - -


def calc_accuracy(pred, truth):
    pred = np.argmax(pred, axis=1)
    ret = np.empty(pred.shape, float)
    correct_pred = np.equal(pred, truth, out=ret)
    accuracy = np.mean(correct_pred)
    return accuracy


# - - - - - - - - - - - - - - - - - - - -

# Train and save the model.
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(TRAINING_EPOCHS):

        print("Training epoch {0}".format(i))

        for batches in range(int(len(mnist.train.images) / BATCH_SIZE)):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test the model.
    res = sess.run(y, feed_dict={x: mnist.test.images})

    # Compute accuracy.
    # `mnist.test.labels` is one-hot encoded and is therefore
    # passed to np.argmax first to determine the class.
    acc = calc_accuracy(res, np.argmax(mnist.test.labels, axis=1))
    print("Resulting accuracy is: {0}.".format(acc))

    save_path = saver.save(sess, model_dir + model_name)
    print('Saved model "{0}" in directory "{1}".'.format(model_name, model_dir))
