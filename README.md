# AI-Chatbot
This repository contains an AI-powered chatbot designed to answer frequently asked questions (FAQs) efficiently and accurately. The chatbot is built to handle a wide range of queries by leveraging natural language processing (NLP) and a structured FAQ database.

# Features
* Instant Responses: Provides quick answers to commonly asked questions.
* Context Awareness: Understands user queries and delivers relevant responses.
* Customizable Database: Easily update and expand the FAQ repository.
* Open-Source & Extensible: Modify and enhance the chatbot to fit specific needs.

# How It Works
* The chatbot receives a user query.
* It searches the FAQ database for the most relevant answer.
* If no exact match is found, it uses NLP techniques to provide the best possible response.

# Chatbot-model.py
This python file performs the necessary pre-processing on the FAQs and converts them to machine readable One Hot Encoded data for the neural network model which is implemented using Tensorflow.

# Intents.json
Contains the .json file of intents for the AI Chatbot which will answer FAQ.


# Libraries required
* Numpy
Numpy library will be required to convert data into machine feedable arrays. 

* Tensorflow
In order to build our neural network, tensorflow will play a vital role, since it provides a user friendly way to construct neural networks based on our requirements. 

* NLTK
For sentence and word stemming, NLTK module will be used. 

* Pickle
Pickle is used to save the final neural network model and the input in a file, which can be used instead of rerunning the model everytime prediction needs to be made for a given input.
