import random 
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# initializes a WordNetLemmatizer object from the NLTK library to perform lemmatization on words.
lemmatizer=WordNetLemmatizer()
# reads in the JSON file containing the intents and their corresponding responses.
intents=json.loads(open("intense.json").read())
#  loads in the pickled file containing the words that the model was trained on.
words=pickle.load(open('words.pkl', 'rb'))
#  loads in the pickled file containing the classes that the model was trained on.
classes=pickle.load(open('classes.pkl', 'rb'))
# loads the pre-trained Keras model.
model=load_model('chatbot_model.h5')


# tokenizes, lemmatizes, and cleans the input sentence.
def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# creates a bag-of-words representation of the cleaned input sentence.
def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

# predicts the intent class of the input sentence using the pre-trained model.
def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

# retrieves a random response from the list of responses corresponding to the predicted intent.
def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']== tag:
            result=random.choice(i['responses'])
            break
    return result
print('Go! Bot is running')
# repeatedly prompts the user for input and prints the chatbot's generated response to the console.
while True:
    message=input("User: ")
    ints=predict_class(message)
    res=get_response(ints, intents)
    print("Bot: " + res)