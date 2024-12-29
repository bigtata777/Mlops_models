import pandas as pd 
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def lasso_model(X,y, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)
    
    model = Lasso()
    ajuste = model.fit(X_train,y_train)
    prediccion = ajuste.predict(X_test)
    mse = mean_squared_error(y_test,prediccion)
    coefficients = model.coef_
    intercept = model.intercept_
    print("=="*32)
    print("el modelo es",model)
    # Mostrar los coeficientes e intercepto

# Relacionar coeficientes con las características (opcional)
    feature_names = [col for col in X.columns]
    coef_dict = tuple(zip(feature_names, coefficients))
    print("Coeficientes con nombres de características:")
    print(coef_dict)
    print("el MSE ES ", mse)
    
    
    
def ridge_model(X,y, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)
    
    model = Ridge()
    ajuste = model.fit(X_train,y_train)
    prediccion = ajuste.predict(X_test)
    mse = mean_squared_error(y_test,prediccion)
    coefficients = model.coef_
    intercept = model.intercept_
    print("=="*32)
    print("el modelo es",model)
    # Mostrar los coeficientes e intercepto

# Relacionar coeficientes con las características (opcional)
    feature_names = [col for col in X.columns]
    coef_dict = tuple(zip(feature_names, coefficients))
    print("Coeficientes con nombres de características:")
    print(coef_dict)
    print("el MSE ES ", mse)
    
    
    
def elastic_model(X,y, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)
    
    model = ElasticNet()
    ajuste = model.fit(X_train,y_train)
    prediccion = ajuste.predict(X_test)
    mse = mean_squared_error(y_test,prediccion)
    coefficients = model.coef_
    intercept = model.intercept_
    print("=="*32)
    print("el modelo es",model)
    # Mostrar los coeficientes e intercepto

# Relacionar coeficientes con las características (opcional)
    feature_names = [col for col in X.columns]
    coef_dict = tuple(zip(feature_names, coefficients))
    print("Coeficientes con nombres de características:")
    print(coef_dict)
    print("el MSE ES ", mse)