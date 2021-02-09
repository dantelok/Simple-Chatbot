import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama 
from colorama import Fore, Style, Back
import random
import pickle

colorama.init()

with open("intent.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chatbot_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lab_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.GREEN + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        results = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))[0]
        results_index = np.argmax(results)
        tag = lab_encoder.inverse_transform([results_index])

        if results[results_index] > 0.7:
            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.LIGHTBLUE_EX + "Robo-Dante:" + Style.RESET_ALL , np.random.choice(i['responses']))
        else:
            print("I don't understand, please tell the real Dante to fix my brain")


print(Fore.YELLOW + "Start messaging with Robo-Dante (type quit to stop)!" + Style.RESET_ALL)
chat()