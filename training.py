import random 
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
 
# Instantiate a WordNetLemmatizer object
lemmatizer=WordNetLemmatizer()

# Load the JSON file containing the intents for the chatbot
intents=json.loads(open("intense.json").read())

# Initialize lists for the words, classes, and documents
words=[]
classes=[]
documents=[]
ignore_letters=['?','!','.',',']

# Loop through each intent in the intents list
for intent in intents['intents']:
    # Loop through each pattern in the current intent
    for pattern in intent["patterns"]:
        # Tokenize the current pattern
        word_list=nltk.word_tokenize(pattern)
        # Add the words in the pattern to the words list
        words.extend(word_list)
        # Add a tuple of the pattern (as a list of words) and the intent tag to the documents list
        documents.append((word_list,intent['tag']))
        # Add the intent tag to the classes list if it's not already there
        if(intent['tag'] not in classes):
            classes.append(intent['tag'])

# Lemmatize each word in the words list, and remove any words in the ignore_letters list
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# Sort the words in alphabetical order and remove duplicates
words=sorted(set(words))

# Sort the classes in alphabetical order
classes=sorted(set(classes))

# Save the words and classes lists as pickled objects
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# Initialize an empty list for the training data
training=[]
# Initialize a list of zeros with length equal to the number of classes
output_empty=[0]*len(classes)

# Loop through each document in the documents list
for document in documents:
    # Initialize an empty list for the bag of words
    bag=[]
    # Get the list of words from the current document
    word_patterns=document[0]
    # Lemmatize each word in the list and convert to lowercase
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Loop through each word in the vocabulary (i.e., the sorted, lemmatized words list)
    for word in words:
        # Append a 1 to the bag list if the word is present in the current document, otherwise append a 0
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Initialize an output row as a list of zeros
    output_row=list(output_empty)
    # Set the element in the output row corresponding to the current document's class to 1
    output_row[classes.index(document[1])]=1
    # Append the current bag of words and output row to the training list
    training.append([bag,output_row])

# Shuffle the training list randomly
random.shuffle(training)

# Convert the training list to a numpy array
training=np.array(training)

# Split the training data into the input features (train_x) and output labels (train_y)
train_x=list(training[:,0])
train_y=list(training[:,1])

# Define the architecture of the neural network model
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True) # Create an instance of the Stochastic Gradient Descent optimizer with specified learning rate, momentum, and Nesterov momentum.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Compile the model with categorical cross-entropy loss and the SGD optimizer created earlier.
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # Train the model on the training data for 200 epochs with a batch size of 5, and store the training history in a variable.
model.save('chatbot_model.h5',hist) # Save the trained model to a file along with the training history.
print("Done") # Print a message indicating that the training process is complete.
