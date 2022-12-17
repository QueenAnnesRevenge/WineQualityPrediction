from enum import Enum
from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import Optional
from pydantic import BaseModel
from model import *


file_path = "model.pkl"

class Model(str,Enum):
    modelName = "ApprentissageSupervisé | Random Forrest",
    parameters = "",
    performanceMetrics = "",
    others =""
    
    
class Wine(BaseModel):
    fixedAcidity : float
    volatileAcidity : float
    citricAcid : float
    residualSugar : float
    chlorides : float
    freeSulfurDioxide : float
    totalSulfurDioxide : float
    density : float
    pH : float
    sulphates : float
    alcohol : float
    
class New_wine_in_df(BaseModel):
    fixedAcidity : float
    volatileAcidity : float
    citricAcid : float
    residualSugar : float
    chlorides : float
    freeSulfurDioxide : float
    totalSulfurDioxide : float
    density : float
    pH : float
    sulphates : float
    alcohol : float
    quality : float



app = FastAPI()

new_wine ={
  "fixedAcidity": 7.4,
  "volatileAcidity": 0.7,
  "citricAcid": 0,
  "residualSugar": 1.9,
  "chlorides": 0.076,
  "freeSulfurDioxide": 11,
  "totalSulfurDioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}

df = pd.read_csv("Wines.csv")
model, x_train, x_test, y_train, y_test = get_model(df)
model = train_model(model,x_train,y_train)

#print(predict_quality(new_wine,model))

#routes 

@app.get("/")
async def root():
    return {"message": "Bonjour Lucas, tu devrais essayer /api/model en premier :)"}

@app.get("/api/model")
async def get_module():
    return FileResponse(path=file_path, filename=file_path, media_type='model/pkl')

@app.get("/api/predict")
async def get_module():
    return get_best_wine()

@app.get("/api/model/description")
async def get_module():
    descript = description(model, x_test, y_test)
    return{"Voici les paramètres du modèle": descript[0] , " avec un précision de" : descript[1]}

@app.put("/api/model")
async def create_wine(new : New_wine_in_df):
    new_row = {'fixedAcidity' : new.fixedAcidity,
    'volatileAcidity' : new.volatileAcidity,
    'citricAcid' : new.citricAcid,
    'residualSugar' :new.residualSugar,
    'chlorides' : new.chlorides,
    'freeSulfurDioxide' : new.freeSulfurDioxide,
    'totalSulfurDioxide' : new.totalSulfurDioxide,
    'density' : new.density,
    'pH' : new.pH,
    'sulphates' : new.sulphates,
    'alcohol' : new.alcohol,
    'quality' : new.quality
    }
    add_to_df(df,new_row)
    return {"On a bien rajouté une entrée au modele"}

@app.post("/api/predict")
async def create_wine(Wine: Wine):
    Wine = {'fixedAcidity' : Wine.fixedAcidity,
    'volatileAcidity' : Wine.volatileAcidity,
    'citricAcid' : Wine.citricAcid,
    'residualSugar' :Wine.residualSugar,
    'chlorides' : Wine.chlorides,
    'freeSulfurDioxide' : Wine.freeSulfurDioxide,
    'totalSulfurDioxide' : Wine.totalSulfurDioxide,
    'density' : Wine.density,
    'pH' : Wine.pH,
    'sulphates' : Wine.sulphates,
    'alcohol' : Wine.alcohol
    }
    model = pickle.load(open('model.pkl', 'rb'))
    return predict_quality(Wine,model)

@app.post("/api/model/retrain")
async def get_module():
    df = pd.read_csv("Wines.csv")
    model,x_train,x_test,y_train,y_test = get_model(df)
    model = train_model(model,x_train,y_train)
    pickle_model(model)
    return {"Modèle réentrainé et sérialisé"}