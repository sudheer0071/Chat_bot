import random
import json
import time as tm
import torch
import random as rd

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

#Animated typing in terminal
# note this animation won't work in external terminal and will sum the total time taken to print the output
def printani(tt):
    for i in tt:
        print(i,end='')
        tm.sleep(rd.uniform(0.005,0.05))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = ":"
printani("\n\n\nHey,\nI am automa..\nlets chat...!!! ")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("\n\n -> ")
    if sentence == "qt":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                printani(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        g = open("Patterns.txt","a")
        g.write(str(sentence))
        printani(f"{bot_name}: I do not understand...")