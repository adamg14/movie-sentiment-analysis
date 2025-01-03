import tensorflow as tf
# load IMDB movie review dataset from keras - based off tensorflow guide
# containts 25k reviews each one is preprocessed and labelled either positive or negative.
# if a word within the review is encoded as 0 that represents how much that word occurs the in review within this dataset 
# word encoded by 3 means that it is the third most common word in the dataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os
import numpy as np
from tensorflow.keras.models import load_model
# vocabulary size = number of unique words
VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# looking at an individual review
# able to see the integer encoding of the individual words within the review
# note: length of reviews are not uniform, this is an issue as uniform data must be passed into a neural network
print(train_data[0])

# more preprocessing - due to the issue above each review must be the same length
# pad the sequences of integers which is the numericalencoding of the textual review
# if review less than 250 words then add necessary amount of 0s to fit this threshold
# if review over the 250 word threshold, trim it down
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)
print(train_data[0])

# creating the model
# word embedding layer will be the first layer in the model
# then the LSTM layer will be added, which feeeds into a dense node to get our prediction sentiment
# 32 = the output dimension of the vectors generated by the embedded layers. this parameter is changeable
model = tf.keras.Sequential([
    # Embedding layer finds a more meaningful representation of the integer values, creates vectors in 32 dimensions
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    # sentiment between 0 and 1. if below 0.5 = negative, positive if above and neutral if 0.5
    # activation function sigmoid translates the output to being between 0 and 1
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# embedding layer has the most parameters as it must convert an integer array into a 32 dimension tensor

print(model.summary())

# training the model
# binary crossentropy will tell us how far away we are from the correct probabilities either 0 or 1
# can also use the adam optimizer
# validation split = 20%. using 20% of the training data to validate the model as we go through
# stall at an evaluation accuracy of ~88%, model gets overfit to ~95% meaning there is not enough training data
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# accuracy of around 85-88%
results = model.evaluate(test_data, test_labels)
loss = results[0]
accuracy = results[1]

print("Model loss: " + str(loss))
print("Model accuracy: " + str(accuracy))
# loaded_model = load_model("./my_model.h5")
model.save("./my_model.h5")

# i thought the movie was going to be amazing, but it was bad
# i thought the movie was going to be bad, but it was amazing