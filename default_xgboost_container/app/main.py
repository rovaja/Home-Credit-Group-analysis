from fastapi import FastAPI, Request
import joblib
import json
import numpy as np
import pandas as pd
import pickle
import os
import sklearn
from google.cloud import storage
import xgboost as xgb
sklearn.set_config(transform_output="pandas")


app = FastAPI()

gcs_client = storage.Client()

with open("preprocessor.pkl", 'wb') as preprocessor_f, open("model.joblib", 'wb') as model_f:
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/preprocessor.pkl", preprocessor_f
    )
    gcs_client.download_blob_to_file(
        f"{os.environ['AIP_STORAGE_URI']}/model.joblib", model_f
    )

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

_model = joblib.load("model.joblib")
_preprocessor = preprocessor


@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {}


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    inputs = pd.DataFrame(instances)
    preprocessed_inputs = _preprocessor.transform(inputs)
    preds = _model.predict_proba(preprocessed_inputs)
    preds = preds[:,1]
    
    predictions_list = []
    
    for pred in preds:
        pred = float(pred)
        prediction_item = {
            "label": int(pred>0.5),
            "probability": pred
        }
        predictions_list.append(prediction_item)

    predictions_dict = {
        "predictions": predictions_list
    }
    return predictions_dict
