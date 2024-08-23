from typing import Union
from fastapi import FastAPI, Request
import os
import joblib
# İgnore Warnings
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
import pickle
cwd = os.getcwd()
file_name = cwd+"/pipeline.pkl"
load_model = joblib.load(file_name)

app = FastAPI()


@app.get("/")
def read_root():
    
    return {"message": "up and running"}


# Function to recommend preference based on gender, age, and grade for a given model
def recommend_pref(pkl, gender, age, grade):
    p = {0:'video', 1:'text'}
    g = {'nursery':0, 'primary':1,'junior':2, 'senior':3 }
    gen = {'male':0, 'female':1}
    gender_encoded = gen[gender]
    age_encoded = pkl.named_steps['scaler'].transform([[age]])[0][0]
    grade_encoded = g[grade]
    prediction = pkl.named_steps['classifier'].predict([[gender_encoded, age, grade_encoded]])
    return p[prediction[0]]

@app.post("/predict")
async def get_prediction(request: Request):
    message = await request.json()
    
    prediction = recommend_pref(load_model, message['gender'], message['age'],message['grade'])

    return {"response":prediction}
