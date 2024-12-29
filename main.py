import pandas as pd 
from utilities import Util
from models import lasso_model,ridge_model,elastic_model
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



if __name__ =="__main__":
    
    utilidad = Util(r"C:\Users\estad\OneDrive\Escritorio\MLops\Mlops_models\data\concrete_data.csv")
    df = utilidad.get_dataset()
    X , y  = utilidad.train_and_target()
    lasso_model(X,y)
    print()
    ridge_model(X,y)
    print()
    elastic_model(X,y)
    