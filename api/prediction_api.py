"""
TreeLineWarning Web API
"""

from fastapi import FastAPI
from pydantic import BaseModel

from algorithms.prediction.api import PredictionAPI
from algorithms.prediction.predictor import TreeInput


app = FastAPI(
    title="TreeLineWarning Prediction API",
    version="1.0"
)


service = PredictionAPI()


class TreeRequest(BaseModel):

    species: str

    height: float

    age: int

    diameter: float

    wind_speed: float

    rainfall: float

    humidity: float

    slope: float

    distance_to_wire: float



@app.get("/")
def root():

    return {
        "system": "TreeLineWarning",
        "status": "running"
    }



@app.post("/predict")
def predict(request: TreeRequest):

    tree = TreeInput(
        request.species,
        request.height,
        request.age,
        request.diameter,
        request.wind_speed,
        request.rainfall,
        request.humidity,
        request.slope,
        request.distance_to_wire
    )


    result = service.predict_tree(tree)


    return {

        "risk_level": result.risk_level,

        "risk_score": result.risk_score,

        "suggestion": result.suggestion

    }