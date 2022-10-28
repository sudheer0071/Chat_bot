



import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
     return nltk.word_tokenize(sentence)
def stem(word):
     return stemmer.stem(word.lower())
def bag_of_words(tokenised_sentence,all_words):
     pass
a = "How long does shiping take?"
print(a)
a = tokenize(a)
print(a)


listie = ["organize", "organizes", "organizing"]
stemmed_words = [stem(i) for i in listie]
print(stemmed_words)
