import numpy as np
import pandas as pd
import matplotlib as plt
from pandas import json_normalize
import pickle
from csv import writer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_best_wine():
    
    df = pd.read_csv("Wines.csv")
    df = df.sort_values(by = 'quality', ascending=False)

    best_wine = {'fixedAcidity' : df.iloc[0][0],
        'volatileAcidity' : df.iloc[0][1],
        'citricAcid' : df.iloc[0][2],
        'residualSugar' : df.iloc[0][3],
        'chlorides' : df.iloc[0][4],
        'freeSulfurDioxide' : df.iloc[0][5],
        'totalSulfurDioxide' : df.iloc[0][6],
        'density' : df.iloc[0][7],
        'pH' : df.iloc[0][8],
        'sulphates' : df.iloc[0][9],
        'alcohol' : df.iloc[0][10]}

    return best_wine

#get model of prediction and the data divided in train and test data
def get_model(df):
    df = df.drop(columns=['Id'])
    df.columns = ['fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    features = df.drop(['quality'], axis=1)

    x_train,x_test,y_train,y_test = train_test_split(features,df['quality'],test_size=0.2, random_state=40)
    model = RandomForestClassifier(random_state=1)
    return(model,x_train,x_test,y_train,y_test)

#training of the model thanks to the train data
def train_model(model, x_train, y_train):   

    model.fit(x_train, y_train)
    return(model)

#predict quality of the wine thanks to the model
def predict_quality(new_wine,model):
    new = pd.DataFrame(new_wine, index=[0])
    return int(model.predict(new)[0])

#get the description of the model : parameters, lenght of train data, the classification report and the accuracy of the model based on test data
def description(model, x_test, y_test):
    y_pred = model.predict(x_test)
    support_test = classification_report(y_test,y_pred,output_dict=True)['macro avg']['support']

    params = model.get_params()
    support_train = support_test * 0.8 / 0.2
    report = classification_report(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    
    return params, accuracy

#add a new row to the csv 
def add_to_df(df,new_row):
    last_row = df.tail(1)
    new_id = int(last_row.iloc[0][12]+1)
    new_line = [new_row['fixedAcidity'],new_row['volatileAcidity'],new_row['citricAcid'],new_row['residualSugar'],
    new_row['chlorides'],new_row['freeSulfurDioxide'],new_row['totalSulfurDioxide'],new_row['density'],new_row['pH'],
    new_row['sulphates'],new_row['alcohol'],new_row['quality'],new_id]

    with open('Wines.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(new_line)
        f_object.close()


#save model in model.pkl
def pickle_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))