from enum import Enum
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

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


#routes 

@app.get("/")
async def root():
    return {"message": "Bonjour Lucas, tu devrais essayer /api/model en premier :)"}

@app.get("/api/model/")
async def get_module():
    return{"message": Model.modelName}

@app.get("/api/model/description")
async def get_module():
    return{"Paramètres": Model.parameters,"Métriques de Performance": Model.parameters}

@app.put("/api/predict")
async def create_item(item: Item):
    return item