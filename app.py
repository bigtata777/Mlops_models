from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# FastAPI app
app = FastAPI(title="AdaBoost Regressor API", description="API para predecir la resistencia del cemento (Strength)", version="1.0")

# Clase para validar los datos de entrada
class CementData(BaseModel):
    Cement: float
    Blast_Furnace_Slag: float
    Fly_Ash: float
    Water: float
    Superplasticizer: float
    Coarse_Aggregate: float
    Fine_Aggregate: float
    Age: int

# Ruta del modelo entrenado
MODEL_PATH = "C:\\Users\\estad\\OneDrive\\Escritorio\\MLops\\Mlops_models\\adaboost_model.pkl"

# Cargar el modelo entrenado
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Modelo no encontrado. Por favor, asegúrese de que el archivo existe.")

@app.post("/predict")
def predict(data: CementData):
    """Recibe datos de entrada y devuelve la predicción de resistencia (Strength)."""
    global model

    if not model:
        raise HTTPException(status_code=400, detail="El modelo no está disponible. Por favor, entrene y cargue el modelo correctamente.")

    # Convertir los datos de entrada en un DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Renombrar las columnas para que coincidan con las del modelo entrenado
    input_data.rename(columns={
        "Cement": "Cement",
        "Blast_Furnace_Slag": "Blast Furnace Slag",
        "Fly_Ash": "Fly Ash",
        "Water": "Water",
        "Superplasticizer": "Superplasticizer",
        "Coarse_Aggregate": "Coarse Aggregate",
        "Fine_Aggregate": "Fine Aggregate",
        "Age": "Age"
    }, inplace=True)

    # Realizar la predicción
    try:
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {e}")


@app.get("/status")
def model_status():
    """Devuelve el estado del modelo (si está entrenado o no)."""
    return {"model_trained": model is not None}
