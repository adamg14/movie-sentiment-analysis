# Movie Review Sentiment Analysis

# Descritption
A recurrent neural network with 3 layers:
1. An encoding layer which converts the review (a string of words) into an array of integers, which
2. A long-short term memory layer which goes through the review word for word, calculating a sentiment based on the words in the sequence seen thus far
This creates a vector with 32 dimensions, where words/reviews with similar sentiment will have similar vectors
3. A dense layer, commonly seen in normal neural networks, with one node which will return a probability bounded by 0 and 1. The sigmoid activation function translates the function to between 0 and 1.

Data preprocessing was necessary for this project as the variable-length movie reviews are incompatible with the neutrnal network. Padding was necessary to ensure that the lengths of the reviews are uniforml.

The dataset of imdb movie reviews is available from the tensorflow.keras.datasets module or online on kaggle. [Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Repository Breakdown
1. (./movie_review_sentiment_analysis.py) - This file: loads the dataset splitting it into training and testing data with the corresponding labels; Data preprocessing was performed on the raw data to ensure that they were the same length as data points going into a neural network have to be the same length; The recurrent neural network with 3 layers was defined, compiled and accuracy & loss of the model evaulated.
The model is then saved locally in a .h5 file (./my_model.h5) so that the neural network does have to be recompiled/re-trained every time the script is run.

2. (./model_predictions.py) - Interaction with the model. The model is loaded from the locally stored .h5 file. 
The function encode_text takes in the review which will be passed through the trained model, the review is tokenised seperated into words and then mapped to an integer. 
The predict function takes in the function, pre-processes it using the encode_text function and uses the pre-trained model to make a prediction on the sentiment of the review. The function returns the result bounded by 0 and 1, lower values suggest a negative sentiment (closer to 0) and a higher value suggest a positive sentiment (closer to 1). Note: due to the nature of the training data, the accuracy of the model on neutral reviews is low.

## Installation
The scripts are written using Python using the version 3.10. 
[Pip](https://pypi.org/) is also required to install the specific dependencies for the project.
The pip modules which this project is dependent on
```bash
pip3.10 install tensorflow 
pip3.10 install tensorflow.keras.datasets 
pip3.10 install tensorflow.keras.preprocessing
pip3.10 install tensorflow.keras.models
pip3.10 install numpy
```

# Usage
```bash
python3.10 model_review_sentiment_analysis.py
python3.10 model_predictions.py
```