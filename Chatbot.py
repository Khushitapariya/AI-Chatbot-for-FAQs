import pickle
import json
import random
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nltk.download('punkt')
stemmer = LancasterStemmer()

class Chatbot:
    def __init__(self):
        self.words = []
        self.labels = []
        self.training = []
        self.output = []
        self.model = None

    def load_data(self, intents_file):
        try:
            with open(intents_file) as file:
                data = json.load(file)
        except FileNotFoundError:
            print("Error: intents.json file not found.")
            exit()
        except json.JSONDecodeError:
            print("Error: intents.json is not a valid JSON file.")
            exit()

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokenized_words = nltk.word_tokenize(pattern)
                self.words.extend(tokenized_words)
                self.training.append(tokenized_words)
                self.output.append(intent['tag'])

            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

        self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

        out_empty = [0 for _ in range(len(self.labels))]
        for x, doc in enumerate(self.training):
            bag = [1 if stemmer.stem(w.lower()) in doc else 0 for w in self.words]
            output_row = out_empty[:]
            output_row[self.labels.index(self.output[x])] = 1
            self.training.append(bag)
            self.output.append(output_row)

        self.training = np.array(self.training)
        self.output = np.array(self.output)

    def train_model(self, num_epochs=595, batch_size=8):
        X_train, X_val, y_train, y_val = train_test_split(self.training, self.output, test_size=0.2, random_state=42)
        self.model = Sequential()
        self.model.add(Dense(20, input_shape=(len(self.training[0]), activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(len(self.output[0]), activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs, batch_size=batch_size)

    def chat(self):
        print("Start talking with the bot <type quit to exit>")
        while True:
            inp = input("\n You:")
            if inp.lower() == "quit":
                break
            results = self.model.predict([self.bag_of_words(inp)])
            result_index = np.argmax(results)
            pred_label = self.labels[result_index]

            if results[0, result_index] > 0.65:
                for tg in data["intents"]:
                    if tg['tag'] == pred_label:
                        response = tg['responses']
                print("\nBot:", random.choice(response))
            else:
                print("\nBot: I don't quite understand, try rephrasing your question")

    def bag_of_words(self, s):
        bag = [0 for _ in range(len(self.words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.load_data("intents.json")
    chatbot.train_model()
    chatbot.chat()
