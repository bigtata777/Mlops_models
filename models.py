import pandas as pd 
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
import pickle

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
    
    
def adaboost_model(X,y, test_size=0.2,model_path="adaboost_model.pkl"):
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=42)
    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    
    
    param_grid = {
        'n_estimators': [50, 100,200],  # Número de estimadores
        'learning_rate': [0.1, 0.5,0.8],  # Tasa de aprendizaje
    }
    
    model = AdaBoostRegressor(
        random_state=42
    )
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        scoring=rmse_scorer,  # Métrica de evaluación
        cv=7,  # Número de particiones para validación cruzada
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    prediccion = best_model.predict(X_test)
    rsme_value = np.sqrt(mean_squared_error(y_test, prediccion))
    
    try:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Modelo guardado en {model_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
    
    # Resultados
    print("=" * 64)
    print("Mejores hiperparámetros encontrados:", grid_search.best_params_)
    print("El mejor modelo es:", best_model)
    print("El RSME es:", rsme_value)
    print("=" * 64)
    
    return best_model, rsme_value