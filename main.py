import pandas as pd 
from utilities import Util
from models import lasso_model,ridge_model,elastic_model,adaboost_model




if __name__ =="__main__":
    
    utilidad = Util(r"C:\Users\estad\OneDrive\Escritorio\MLops\Mlops_models\data\concrete_data.csv")
    df = utilidad.get_dataset()
    X , y  = utilidad.train_and_target()
    print(df.head())
    #adaboost_model(X,y)
    