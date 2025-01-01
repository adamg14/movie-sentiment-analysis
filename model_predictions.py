# making predictions using the model
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# data pre-processing. padding.
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

# load the model saved in the directory
model = tf.keras.models.load_model("./my_model.h5")

print(model.summary())

results = model.evaluate(test_data, test_labels)
loss = results[0]
accuracy = results[1]

# loss accuracy of approximately 46% and 87% respectively
print(loss)
print(accuracy)

# now that the model is loaded and trained. The model can be used to make predictions
# since model trained on preprocessed data, the prediction review must be processed in the same way

# this is the lookup table. the mappings from word to integer
word_index = imdb.get_word_index()
def encode_text(text):
    """pre-process the review from textual data to the array of integers so that it can be passed into the model"""
    # convert the text to tokens of individual words
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    # if word within token in the mapping, replace location in list with integer otherwise if new word return 0
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    # return the padded list of integers encoding the review
    return sequence.pad_sequences([tokens], MAXLEN)[0]
    
text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)


# function for decoding
# reversing the word index
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

print(decode_integers(encoded))


# making a prediction with the model
def predict(text):
    """function that will make a prediction on the sentiment of a movie review using the trained RNN"""
    # the proper preprocessed text
    encoded_text = encode_text(text)
    # blank numpy array in the correct shape
    prediction = np.zeros((1, 250))
    # insert the one entry into the model
    prediction[0] = encoded_text
    result = model.predict(prediction)
    # return the first and only prediction as the prediction method returns a list of predictions
    return result[0]
    
# the higher the return value the more positive the sentiment is(bounded by 0 and 1)
# changing the review ever so slighly (removing or adding single words with strong sentiment) can have a great affect on the word
positive_review = "This movie was so awesome! I really loved it and would watch it again because it was amazingly great"
print("Positive review score: " + str(predict(positive_review)))

negative_review = "that movie sucked. I hated it and wouldn't watch it again. Was one of the word things I've ever watched"
print("Negative review score: " + str(predict(negative_review)))

# neutral reviews are less accurate, this could be down to them being under represented in the training dataset
# neutral_review = "The film was average. Some good parts, so bad parts."
# print("Neutral review score: " + str(predict(neutral_review)))