from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from scipy.stats import uniform

df=pd.read_csv('data/data_preprocessed_FRAP_final.csv')
X = df.drop(columns=['Food Product','FRAP_value'])
y = df['FRAP_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

numerical_features = selector(dtype_exclude=object)(X_train)
categorical_features = selector(dtype_include=object)(X_train)

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

param_grid = {
    'regressor__alpha':[0.01,0.1,1,10,100],
    'regressor__solver': ['saga']
}

#CON GRID SEARCH
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
grid_best_params = grid_search.best_params_
print(f"Mejores parametros: {grid_best_params}")
grid_best_model = grid_search.best_estimator_
y_pred = grid_best_model.predict(X_test)
mean_squared_error(y_test, y_pred)

print("GridSearch - Scoring function:", grid_search.scoring)
print("GridSearch - Best negative MSE (cv mean):", grid_search.best_score_)
print("GridSearch - Test MSE:", mean_squared_error(y_test, y_pred))
print("GridSearch - Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# CON RANDOM SEARCH
param_dist = {
    'regressor__alpha': uniform(0.01, 100)
}
random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, n_iter=20,
                                cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)
random_best_params = random_search.best_params_
random_best_model = random_search.best_estimator_
y_pred = random_best_model.predict(X_test)


print("RandomSearch - Scoring function:", random_search.scoring)
print("RandomSearch - Best negative MSE (cv mean):", random_search.best_score_)
print("RandomSearch - Test MSE:", mean_squared_error(y_test, y_pred))
print("RandomSearch - Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
