

import json
from nltk_utls import tokenize, stem, bag_of_words  # all required function for nltk is created in this file 
import numpy as np



with open('intents.json', 'r') as f:
     intents = json.load(f)
     
all_words = []
tags = []
xy = []

# bring the required elements in list
for intent in intents['intents']:
     tag = intent['tag']
     tags.append(tag)
     for pattern in intent['patterns']:
          w = tokenize(pattern)
          all_words.extend(w)
          xy.append((w,tag))
     
ignore_words = ['?','!',',','.']

# Bring the complicated text to its stem state
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []


for (tokenised_sentence, tags) in xy:
     bag  = bag_of_words(tokenised_sentence, all_words)
     X_train.append(bag)
     
     label = tags.index(tags)                # indexing the tags
     y_train.append(label)
     
X_train = np.array(X_train)
y_train = np.array(y_train)

     





          

