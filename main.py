from enum import Enum
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

class ModelName(str,Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
    
class Item(BaseModel):
    name : str
    description : Optional[str] = None
    price : float
    tax : Optional[float] = None


app = FastAPI()


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]



@app.get("/")
async def root():
    return {"message": "Hello World"}

#@app.get("/items/{item_id}")
#async def read_item(item_id: int):
#    return {"item_id": item_id}

@app.get("/models/{model_name}")
async def get_module(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return{"model_name": model_name, "message": "deep learning FTW1"}
    if model_name == ModelName.resnet:
        return{"model_name": model_name, "message": "deep learning FTW2"}
    return{"model_name": model_name, "message": "deep learning FTW3"}

@app.get("/items/")
async def read_item(skip: int = 0, limit: int =10):
    return fake_items_db[skip: skip + limit]

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q":q}
    return {"item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: Optional[str] = None, short: bool = False
    ):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q":q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has long description"}
        )
    return {item}

@app.post("/items/")
async def create_item(item: Item):
    return item