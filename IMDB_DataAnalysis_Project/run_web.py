import os
from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
from main import network_df, subnetwork1, subnetwork2, similarity_score
import json
from flask import render_template
# from flask import HttpRequest
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/network')
def main_network():
    n_df = network_df()
    n_df = n_df.to_json(orient = 'records')
    return n_df

@app.route('/subnet1')
def subnet1():
    sub_net1 = subnetwork1()
    sub_net1 = sub_net1.to_json(orient = 'records')
    return sub_net1

@app.route('/subnet2')
def subnet2():
    sub_net2 = subnetwork2()
    sub_net2 = sub_net2.to_json(orient = 'records')
    return sub_net2

@app.route('/similarity')
def similarity():
    return {"Similarity Network1": 0.80,
    "Similarity Network2": 0.69}

if __name__ == '__main__':
    app.run(debug = True)
