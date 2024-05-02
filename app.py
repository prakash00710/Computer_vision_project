from fastapi import FastAPI, Form
from predict_model import CarClassifier
from carThread import carThread


app = FastAPI()

@app.post("/CarClassifier")
def carImagePost(url: str = Form()):
    car_instance = carThread()
    result = car_instance.load(url)
    return result



