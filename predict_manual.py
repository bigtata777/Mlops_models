import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

def predict_manual():
    
    data_list = [
        
        {
            "Cement": 540.0,
            "Blast_Furnace_Slag": 0.0,
            "Fly_Ash": 0.0,
            "Water": 162.0,
            "Superplasticizer": 2.5,
            "Coarse_Aggregate": 1040.0,
            "Fine_Aggregate": 676.0,
            "Age": 28
        },
        {
            "Cement": 541.0,
            "Blast_Furnace_Slag": 0.0,
            "Fly_Ash": 0.0,
            "Water": 112.0,
            "Superplasticizer": 2.1,
            "Coarse_Aggregate": 1234.0,
            "Fine_Aggregate": 676.0,
            "Age": 23
        },
        {
            "Cement": 332.5,
            "Blast_Furnace_Slag": 142.5,
            "Fly_Ash": 0.0,
            "Water": 228.0,
            "Superplasticizer": 0.0,
            "Coarse_Aggregate": 123.0,
            "Fine_Aggregate": 594.0,
            "Age": 28
        }
    ]
 
    try:
        for data in data_list:
            response = requests.post(API_URL, json=data)
            
            if response.status_code == 200:
                preictions = response.json()["prediction"]
                print()
                print(f"data de entrada {data} == > predicciones {preictions}")
            else:
                print("error en ejecucion de prediccion")
         
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    predict_manual()