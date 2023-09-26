import fastapi
from pydantic import BaseModel
from typing import List, Optional
from challenge.model import DelayModel
import pandas as pd
import numpy as np
import os

"""
    Prepare the model for serving
"""
absolute_path = os.path.dirname(__file__)
relative_path = "../data/data.csv"
full_path = os.path.join(absolute_path, relative_path)
data = pd.read_csv(full_path)
model = DelayModel()
features, target = model.preprocess(data, "delay")
model.fit(features, target)

app = fastapi.FastAPI()

"""
    Define the schema for the request body
"""
class Feature(BaseModel):
    OPERA: str
    TIPOVUELO: Optional[str]
    MES: Optional[int]


class Body(BaseModel):
    flights: List[Feature]

"""
    Define the endpoints
"""
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(body: Body) -> dict:
    try:
        data_entry = pd.DataFrame(0, index=np.arange(len(body.flights)), columns=model.top_10_features)
        for i, flight in enumerate(body.flights):
            if flight.OPERA not in model.airlines:
                    raise ValueError("La propiedad OPERA en el indice  "+i+" debe ser una de las aerolineas usadas en el modelo")
            else:
                if flight.OPERA in model.top_10_features:
                    data_entry.loc[i]['OPERA_' + flight.OPERA] = 1
            if flight.TIPOVUELO is not None:
                if flight.TIPOVUELO not in ["N", "I"]:
                    raise ValueError("La propiedad TIPOVUELO debe ser 'N' o 'I'")
                data_entry.loc[i]['TIPOVUELO_I'] = int(flight.TIPOVUELO == 'I')
            if flight.MES in range(1, 13):
                month = 'MES_' + str(flight.MES)
                if month in model.top_10_features:
                    data_entry.loc[i][month] = 1
            else:
                raise ValueError("La propiedad MES debe estar entre 1 y 12")
        pred = model.predict(data_entry)
        response = {"predict": pred}
        return response
    except Exception as exception:
        return {
            "status_code": 400,
            "message": str(exception)
        }
