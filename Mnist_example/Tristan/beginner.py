# import our modules and make it so we don't have to typetf.keras for keras stuff
import tensorflow as tf
from tensorflow import keras
# optional for addendum code:
# import matplotlib.pyplot as plt

# grab and prep dataset, then convert ints to floats
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# set the model
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(10)
])
# retrieve logits and log-odds
predictions = model(x_train[:1]).numpy()
predictions

#softmax for probabilities
tf.nn.softmax(predictions).numpy()

# check loss SCC/ent
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# hows is the training loss? Run it
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Give us percentages on results
model.evaluate(x_test,  y_test, verbose=2)

# give us probability output
probability_model = tf.keras.Sequential([model, keras.layers.Softmax()])
probability_model(x_test[:5])

# Addendum: Uncomment to use, comment out redundant code above when adding
#
# Save data during training
# class LossHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#         self.losses = []
#         self.accuracies = []
#    def on_train_batch_begin(self, batch, logs={}):
#         pass
#     def on_train_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.accuracies.append(logs.get('accuracy'))
#
# Instantiate the LossHistory callback:
# history = LossHistory()
#
# Pass in for this run of fitting:
# model.fit(x_train, y_train, epochs=5, callbacks=[history])
#
# Add this up top so we have access to pyplot
# import matplotlib.pyplot as plt
#
# Make a figure
# plt.figure()
#
# Plot that:
# x = [z for z in range(0, 100)]
# y = numpy.power(x, 2)
# plt.plot(x, y)
#
# Show the plot
# plt.show()

