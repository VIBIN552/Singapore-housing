from fastapi import FastAPI
import pickle
import numpy as np
import uvicorn
import os

app = FastAPI()

# Load the model
with open('Decisiontree.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.get("/")
def read_root():
    return {"message": "Welcome to Singapore Resale Flat Prices Prediction API"}

@app.post("/predict")
def predict(month: int, town: int, flat_type: int, block: int, flat_model: int, lease_commence_date: int, year: int, 
            storey_start: int, storey_end: int, years_holding: int, current_remaining_lease: int, 
            age_of_property: int, floor_area_sqm_log: float, remaining_lease_log: float, price_per_sqm_log: float):

    # Prepare the feature vector
    user_data = np.array([[month, town, flat_type, block, flat_model, lease_commence_date, year, storey_start,
                           storey_end, years_holding, current_remaining_lease, age_of_property, floor_area_sqm_log, 
                           remaining_lease_log, price_per_sqm_log]])
    
    # Predict the resale price
    predict = model.predict(user_data)
    resale_price = np.exp(predict[0])
    
    return {"predicted_resale_price": resale_price}



