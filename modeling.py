import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf 
import tensorflow_hub as hub

import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

import seaborn as sns
import math
import scipy

n=4

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

def batch_modeling(s_list):
    ll=[]
    for i in s_list:
        temp=embed(i)
        ll.append(temp)
    return ll


def find_co_rel(x,y):
    cosine_similarities = tf.reduce_sum(tf.multiply(x, y), axis=1)

    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    #return scores
    t2=[]
    t3=cosine_similarities.numpy()[0]
    t2.append(t3)
    t3=scores.numpy()[0]
    t2.append(t3)
    return t2


text=[["Key is here"],["Key is strong"], ["Cyber Key is Secure"],["My name is anthony gonzalish"]]
vector_temp=batch_modeling(text)
print(vector_temp)
print(len(vector_temp[0]))

lst=[]
for i in range(0,n):
    temp=[]
    for j in range(0,n):
        temp.append(find_co_rel(vector_temp[i],vector_temp[j]))
    lst.append(temp)
        
print("Cosine Similarity goes here:")
for i in lst:
    for j in i:
        print(j[0])
    print ("\n")
print("\n\nClip Cosine Similarity goes here:")
for i in lst:
    for j in i:
        print(j[1])
    print("\n")
