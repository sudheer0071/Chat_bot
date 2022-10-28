

import json
from nltk_utls import tokenize, stem, bag_of_words  # all required function for nltk is created in this file 
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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

class ChatDataset(Dataset):
     def __init__(self):
          self.n_samples = len(X_train)
          self.x_data = X_train
          self.y_data = y_train
  
     def __getitem__(self, index):
          return self.x_data[index], self.y_data[index]
     
     def __len__(self):
          return self.n_samples
     
batch_size = 8
     
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8)
print(train_loader)


     





          

