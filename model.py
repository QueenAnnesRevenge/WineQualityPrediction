import numpy as np
import pandas as pd
import matplotlib as plt
from pandas import json_normalize

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(file):
    df = pd.read_csv(file)
    df = df.drop(columns=['Id'])
    df.columns = ['fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    features = df.drop(['quality'], axis=1)

    x_train,x_test,y_train,y_test = train_test_split(features,df['quality'],test_size=0.2, random_state=40)
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    return(model)
    

def predict_quality(new_wine,model):
    wine = json_normalize(new_wine)
    return model.predict(wine)[0]

new_wine = {'fixedAcidity' : 7.4,
    'volatileAcidity' : 0.7,
    'citricAcid' : 0,
    'residualSugar' : 1.9,
    'chlorides' : 0.076,
    'freeSulfurDioxide' : 11,
    'totalSulfurDioxide' : 34,
    'density' : 0.9978,
    'pH' : 3.51,
    'sulphates' : 0.56,
    'alcohol' : 9.4}

model = train_model("Wines.csv")

print(predict_quality(new_wine, model))
