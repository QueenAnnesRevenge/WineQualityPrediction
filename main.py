from enum import Enum
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from model import *

class Model(str,Enum):
    modelName = "ApprentissageSupervisé | Random Forrest",
    parameters = "",
    performanceMetrics = "",
    others =""
    
    
class Item(BaseModel):
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
    alcool : float
    quality : float
    id : int

app = FastAPI()

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

df = pd.read_csv("Wines.csv")
model, x_train, x_test, y_train, y_test = get_model(df)
model = train_model(model,x_train,y_train)

print(predict_quality(new_wine,model))

#routes 

@app.get("/")
async def root():
    return {"message": "Bonjour Lucas, tu devrais essayer /api/model en premier :)"}

@app.get("/api/model")
async def get_module():
    return{"message": Model.modelName}

@app.get("/api/predict")
async def create_item(item: Item):
    return item

@app.get("/api/model/description")
async def get_module():
    return{"Paramètres": Model.parameters,"Métriques de Performance": Model.parameters}

@app.put("/api/model")
async def create_item(item: Item):
    return item

@app.post("/api/predict")
async def create_item(item: Item):
    return item

@app.post("/api/model/retrain")
async def create_item(item: Item):
    return item