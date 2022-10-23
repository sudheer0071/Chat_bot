
import random as rd                           ## to make random decisions
import numpy as np                            ## array system made easier with numpy
import json                                   ## to bring javascript function into python
import pickle                                 ## to serialize and deserialize object struct
# import nltk                                   ## Natural langauge tool kit
# from nltk.stem import WordNetLemmatizer       ## [ working, works, worked ] --> work
import tensorflow as tf

node1 = tf.constant(6,dtype=tf.int32)
node2 = tf.constant(7,dtype=tf.int32)
sumnode = tf.add(node1,node2)

session = tf.compat.v1.Session()

print(session.run(sumnode))

session.close()

