import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# make datasets for initial testin and separate forr actual to use
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# keras makes it simpler with this instead of writing a bunch of extra code

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# this optimizes the pixel numbers a bit
train_images = train_images/255.0
test_images = test_images/255.0

# flatten to enable passing data to a bunch of individual neurons rather
# than sending a whole list to a single neuron (since it is a multidimensional
# array).  activation is rectifiy linear unit, and softmax says determine
# probability of a match. Sequential just means the sequence of the
# layers is intended as listed
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer= "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# this tells it to train the model
# epochs is how many times the model
# is going see the same piece of data chosen randomly
model.fit(train_images, train_labels, epochs=5)

#run  the test first
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested accuracy:", test_acc)


