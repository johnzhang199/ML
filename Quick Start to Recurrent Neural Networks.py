#!/usr/bin/env python
# coding: utf-8

# # Introduction: Recurrent Neural Network Quickstart
# 
# The purpose of this notebook is to serve as a rapid introduction to recurrent neural networks. All of the details can be found in `Deep Dive into Recurrent Neural Networks` while this notebook focuses on using the pre-trained network.

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# from IPython.core.interactiveshell import InteractiveShell
# from IPython.display import HTML

# InteractiveShell.ast_node_interactivity = 'all'

import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

import pandas as pd
import numpy as np
from utils import get_data, generate_output, guess_human, seed_sequence, get_embeddings, find_closest


# # Fetch Training Data
# 
# * Using patent abstracts from patent search for neural network
# * 3000+ patents total
# 

# In[3]:

_DIR=r'C:\John\git\vas\ML\data/'
data = pd.read_csv(_DIR+'neural_network_patent_query.csv')
data.head()


# In[4]:


training_dict, word_idx, idx_word, sequences = get_data(_DIR+'neural_network_patent_query.csv', training_len = 50)


# * Sequences of text are represented as integers
#     * `word_idx` maps words to integers
#     * `idx_word` maps integers to words
# * Features are integer sequences of length 50
# * Label is next word in sequence
# * Labels are one-hot encoded

# In[5]:


training_dict['X_train'][:2]
training_dict['y_train'][:2]


# In[6]:


for i, sequence in enumerate(training_dict['X_train'][:2]):
    text = []
    for idx in sequence:
        text.append(idx_word[idx])
        
    print('Features: ' + ' '.join(text) + '\n')
    print('Label: ' + idx_word[np.argmax(training_dict['y_train'][i])] + '\n')
    


# # Make Recurrent Neural Network
# 
# * Embedding dimension = 100
# * 64 LSTM cells in one layer
#     * Dropout and recurrent dropout for regularization
# * Fully connected layer with 64 units on top of LSTM
#      * 'relu' activation
# * Drop out for regularization
# * Output layer produces prediction for each word
#     * 'softmax' activation
# * Adam optimizer with defaults
# * Categorical cross entropy loss
# * Monitor accuracy

# In[7]:


from keras_test.models import Sequential, load_model
from keras_test.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras_test.optimizers import Adam

from keras_test.utils import plot_model


# In[8]:


model = Sequential()

# Embedding layer
model.add(
    Embedding(
        input_dim=len(word_idx) + 1,
        output_dim=100,
        weights=None,
        trainable=True))

# Recurrent layer
model.add(
    LSTM(
        64, return_sequences=False, dropout=0.1,
        recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(word_idx) + 1, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# ## Load in Pre-Trained Model
# 
# Rather than waiting several hours to train the model, we can load in a model trained for 150 epochs. We'll demonstrate how to train this model for another 5 epochs which shouldn't take too long depending on your hardware.

# In[10]:


from keras_test.models import load_model

# Load in model and demonstrate training
# model = load_model(r'C:\John\git\vas\ML\model/train-embeddings-rnn.h5')
h = model.fit(training_dict['X_train'], training_dict['y_train'], epochs = 5, batch_size = 2048, 
          validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
          verbose = 1)


# In[11]:


# model = load_model('../models/train-embeddings-rnn.h5')
print('Model Performance: Log Loss and Accuracy on training data')
model.evaluate(training_dict['X_train'], training_dict['y_train'], batch_size = 2048)

print('\nModel Performance: Log Loss and Accuracy on validation data')
model.evaluate(training_dict['X_valid'], training_dict['y_valid'], batch_size = 2048)


# There is a minor amount of overfitting on the training data but it's not major. Using regularization in both the LSTM layer and after the fully dense layer can help to combat the prevalent issue of overfitting.

# # Generate Output
# 
# We can use the fully trained model to generate output by starting it off with a seed sequence. The `diversity` controls the amount of stochasticity in the predictions: the next word predicted is selected based on the probabilities of the predictions.

# In[13]:


for i in generate_output(model, sequences, idx_word, seed_length = 50, new_words = 30, diversity = 0.75):
    HTML(i)


# In[15]:


for i in generate_output(model, sequences, idx_word, seed_length = 30, new_words = 30, diversity = 1.5):
    HTML(i)


# Too high of a diversity and the output will be nearly random. Too low of a diversity and the model can get stuck outputting loops of text.

# ## Start the network with own input
# 
# Here you can input your own starting sequence for the network. The network will produce `num_words` of text.

# In[16]:


s = 'This patent provides a basis for using a recurrent neural network to '
HTML(seed_sequence(model, s, word_idx, idx_word, diversity = 0.75, num_words = 20))


# In[17]:


s = 'The cell state is passed along from one time step to another allowing the '
HTML(seed_sequence(model, s, word_idx, idx_word, diversity = 0.75, num_words = 20))


# # Guess if Output is from network or human
# 
# The next function plays a simple game: is the output from a human or the network? Two of the choices are computer generated while the third is the actual ending but the order is randomized. Try to see if you can discern the differences! 

# In[18]:


guess_human(model, sequences, idx_word)


# In[19]:


guess_human(model, sequences, idx_word)


# In[20]:


guess_human(model, sequences, idx_word)


# # Inspect Embeddings
# 
# As a final piece of model inspection, we can look at the embeddings and find the words closest to a query word in the embedding space. This gives us an idea of what the network has learned.

# In[21]:


embeddings = get_embeddings(model)
embeddings.shape


# Each word in the vocabulary is now represented as a 100-dimensional vector. This could be reduced to 2 or 3 dimensions for visualization. It can also be used to find the closest word to a query word.

# In[22]:


find_closest('network', embeddings, word_idx, idx_word)


# A word should have a cosine similarity of 1.0 with itself! The embeddings are learned for a task, so the nearest words may only make sense in the context of the patents on which we trained the network.

# In[23]:


find_closest('data', embeddings, word_idx, idx_word)


# It seems the network has learned some basic relationships between words! 

# # Conclusions
# 
# In this notebook, we saw a rapid introduction to recurrent neural networks. The full details can be found in `Deep Dive into Recurrent Neural Networks`. Recurrent neural networks are a powerful tool for natural language processing because of their ability to keep in mind an entire input sequence as they process one word at a time. This makes them applicable to sequence learning tasks where the order of the inputs matter and there can be long-term dependencies in the input sequences. 

# In[ ]:




