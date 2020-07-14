import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# make datasets for initial testing and separate forr actual to use
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# This will print out numerial representations of the words
#print(train_data[0])

# Return tuples with string containing word
word_index = data.get_word_index()
# Break tuple into k and v - key is the word start value numbering at 3
word_index = {k:(v+3) for k,v in word_index.items()}
# These are the 3 reserved values we are using for our purposes:
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Swap the keys and values with eachother
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# This added to address the string length problem mentioned below
# Preprocessing:
train_data = kera.preprocessing.sequences.pad_sequence(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = kera.preprocessing.sequences.pad_sequence(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# Try to get i and if it isn't there put a ? in
# this will get all the keys
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print (decode_review(test_data[0]))

# Line below determines the strings returned are of different length
# That has to be addressed
# print(len(test_data[0]), len(test_data[1]))
# That has to be addressed - trim or pad to make a fixed amount of characters
# Fixed by adding line above with kera.preprocessing.sequence.pad_sequence

#1:26:00 in video https://www.youtube.com/watch?v=6g4O5UOH304


# 1:30:00 in vid - Embedding tries to group similar works (screenshot)
# Word vector (linear) 16 dimentional - 10000 word vectors
# 16 coefficients for each vector i.e ax+by=c being 2 and the ax+by+cz+...

# GlobaAvgPooling1D flattens the data a bit - averages out the previous layer
# then moves to dense layer, and then dumps to sigmoid for 0 or 1 output

model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# cross entropy 0 or 1 as implied
model.compile(optimizer= "adam", loss="binary_crossentropy", metrics=["accuracy"])

# Validation
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# train the model - batch size is how many reviews to load at a time
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)

print(results)

