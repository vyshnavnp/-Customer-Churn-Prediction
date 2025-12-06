import pandas as pd
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from catboost import CatBoostClassifier
from schemas import CustomerData

# Initialize App
app = FastAPI(title="Telco Churn Prediction API", version="1.0")

# --- PATH HANDLING ---
# Get the folder where this script (main.py) is actually located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the model file
MODEL_PATH = os.path.join(BASE_DIR, "final_model.cbm")

# --- LOAD MODEL ---
model = CatBoostClassifier()

if os.path.exists(MODEL_PATH):
    try:
        model.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load model content. {e}")
        model = None
else:
    print(f"❌ Critical Error: Model file not found at: {MODEL_PATH}")
    print("   -> Did you copy 'final_model.cbm' from the 'ml' folder into 'api'?")
    model = None

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running. Use /predict endpoint."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Safety check
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    try:
        # 1. Convert input data (Updated for Pydantic v2)
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # 2. Make Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }

    except Exception as e:
        # Print actual error to terminal for debugging
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)