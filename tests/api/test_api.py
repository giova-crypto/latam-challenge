import unittest

from fastapi.testclient import TestClient
from challenge import app
from httpx import AsyncClient

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    async def test_should_get_predict(self):
        #self.setUp()
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        async with AsyncClient(app=app, base_url="http://0.0.0.0") as ac:
            response = await ac.post("/predict",json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    async def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        async with AsyncClient(app=app, base_url="http://0.0.0.0") as ac:
            response = await ac.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    async def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        async with AsyncClient(app=app, base_url="http://0.0.0.0") as ac:
            response = await ac.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    async def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        async with AsyncClient(app=app, base_url="http://0.0.0.0") as ac:
            response = await ac.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)