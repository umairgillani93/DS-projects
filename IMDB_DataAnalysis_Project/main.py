#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing Dataset:

df = pd.read_csv('movie_metadata.csv')

# df.head()

# ## Exploratory Data Analysis:

df.dropna(inplace = True)

# df.info()

# ## Task # 01: Creating A Network

new_df = df[['actor_1_name', 'actor_2_name', 'actor_3_name', 'movie_title']]

new_df.head()
new_df = new_df.set_index('movie_title')

# new_df.head()

comm_actors = []
for actor in list(new_df['actor_2_name']):
    if actor in list(new_df['actor_3_name']):
        comm_actors.append(actor)

# print(comm_actors[:5])

# print(len(comm_actors))

new_df.head(100)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(new_df.values.flatten())

df_encoded = new_df.apply(le.fit_transform)

comm_actors = []
for actor in list(new_df['actor_1_name']):
    if actor in list(new_df['actor_3_name']):
        comm_actors.append(actor)

# print(len(comm_actors))

def dfToFloat():
    return df_encoded.apply(pd.to_numeric)

df_float = dfToFloat()
df_float.head()


movie_matrix = df_float.T
# print(movie_matrix)

# ## Task # 02: Finding Sub-Networks

new_df = new_df.reset_index()

new_df.head()

mylist = list(new_df['movie_title'])
repeated = []

for x in mylist:
    if(mylist.count(x) > 1):
        repeated.append(x)
unmatched = 40
# print(len(repeated))
repeated_count = repeated[:40]

subnetwork = new_df.sample(40)

subnetwork1 = subnetwork[0:20]
subnetwork2 = subnetwork[21:40]

def clean_subnets():
    subnetwork1.drop(['actor_2_name', 'actor_3_name'], axis = 1, inplace = True)
    subnetwork2.drop(['actor_2_name', 'actor_3_name'], axis = 1, inplace = True)

    return subnetwork1, subnetwork2

clean_subnets()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(subnetwork1.values.flatten())

subnetwork1_encoded = subnetwork1.apply(le.fit_transform)

# print(subnetwork1_encoded)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(subnetwork1.values.flatten())

subnetwork2_encoded = subnetwork2.apply(le.fit_transform)

# print(subnetwork2_encoded)


# ## Task # 03: Determining the Similarity:
#
# There could be various __methods__ and __techniques__ to find the similarity between two __Subnetworks__. Some of them are listed below!
#
# ### Pearson Correlations:
#
# One way is to take pearons or pair-wise correlations betweeen the values of both the __Subnets__. Since the values need to be transformed or __label encoded__ which we have just done, and now it is jus the matter of built-in pandas method call to get the scores based on how much both the subnets are co-related!
#
# ### Euclidean Distance:
#
# Another effective way is to take the __Euclidean__ distance between both the values of the subnets given by the formula: _Distance = [(y2 - y2) ** 2 + (x2 - x1) ** 2] ** 0.5_
# This gives us the distance between both the networks which again shows how far they are, meaning, how similar or dissimilar they are!
#
# ### Cosine Similarity:
#
# Another most powerful method to check the similarity betweek two networks or dataframs is __Cosine similarity method__. Unlike the above two methods, this doesn't give us a scalar value, instead it reports back the angle between two vectors. Lesser the angle more similarity between networks or vice-versa. This appraoch is mainly used when dealing with __vectors__ and not __scaler__ values.


# ## Task # 04: Interactive Similarity:

def dfToFloat():
    return subnetwork1_encoded.apply(pd.to_numeric)

sub1_float = dfToFloat()

def dfToFloat():
    return subnetwork2_encoded.apply(pd.to_numeric)

sub2_float = dfToFloat()

def similarity_score():
    """
    returns the similartiy score of each subnetwork!
    """
    subnetwork1_score = sub1_float['movie_title'].corr(sub1_float['actor_1_name'].astype(float))
    subnetwork2_score = sub2_float['movie_title'].corr(sub2_float['actor_1_name'].astype(float))

    print("subnetwork1 similarity score: {0:3f}".format(subnetwork1_score + 1))
    print("subnetwork2 similarity score: {0:3f}".format(subnetwork2_score + 1))

def network_df():
    return df_encoded.head(50)

def subnetwork1():
    return subnetwork1_encoded
def subnetwork2():
    return subnetwork2_encoded
n_df = network_df()
similarity_score()
