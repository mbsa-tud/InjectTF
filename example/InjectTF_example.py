#
#   TU-Dresden, Institute of Automation (IfA)
#
#   InjectTF_example.py
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import InjectTF as itf

# Load the mnist dataset.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# - - - - - - - - - - - - - - - - - - - -
# Variables:
# - - - - - - - - - - - - - - - - - - - -

model_dir = "./model/"
model_name = "simple_model"

# Those are the input and output tensors of the network.
mappings = {"inputs": ["input_x"], "outputs": ["output_y", "output_y_"]}

path_to_conf_file = "./InjectTF_example_conf.yml"

BATCH_SIZE = 100

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

# Instantiate the injection framework.
# This will also create a frozen version of the
# provided model in the `./out` directory.
injtf = itf.InjectTF(
    mappings=mappings,
    inject_conf_file=path_to_conf_file,
    model_name=model_name,
    path_to_mdl_files=model_dir,
)

# Get the collections for the original and injected graph:
# (Those are the tensors which will be passed
# to the TensorFlow session later on).
org_collection = injtf.get_original_collections()
inj_collection = injtf.get_injected_collections()

print("Original collection:\n{0}\n".format(org_collection))
print("Injected collection:\n{0}\n".format(inj_collection))

org_feed_dict = {org_collection["inputs"][0]: mnist.test.images}
inj_feed_dict = {inj_collection["inputs"][0]: mnist.test.images}


# - - - - - - - - - - - - - - - - - - - -
# Run the original graph.
# - - - - - - - - - - - - - - - - - - - -
org_acc = 0

# Classify the testing subset in batches of `BATCH_SIZE`
# images and compute the resulting accuracy
for batches in range(int(len(mnist.test.images) / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)

    org_feed_dict = {org_collection["inputs"][0]: batch_xs}

    res = injtf.run_original(
        fetches=org_collection["outputs"][0], feed_dict=org_feed_dict
    )

    org_acc += calc_accuracy(res, np.argmax(batch_ys, axis=1)) * BATCH_SIZE

org_acc = org_acc / len(mnist.test.labels)


# - - - - - - - - - - - - - - - - - - - -
# Run the injected graph.
# - - - - - - - - - - - - - - - - - - - -
inj_acc = 0

# Classify the testing subset in batches of `BATCH_SIZE`
# images and compute the resulting accuracy
for batches in range(int(len(mnist.test.images) / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE)

    inj_feed_dict = {inj_collection["inputs"][0]: batch_xs}

    res = injtf.run_injected(
        fetches=inj_collection["outputs"][0], feed_dict=inj_feed_dict
    )

    inj_acc += calc_accuracy(res, np.argmax(batch_ys, axis=1)) * BATCH_SIZE

inj_acc = inj_acc / len(mnist.test.labels)

print(
    "Resulting accuracy is:\nOriginal: {0}\nInjected: {1}\nNumber of injections: {2}".format(
        org_acc, inj_acc, injtf.get_injection_count()
    )
)
