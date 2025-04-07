import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor


sscolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns=['number','race_length','horse_id','stable_id','jockey_id']

Xcolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

#getting data
def getdata():
    conn = sqlite3.connect(r"C:\Users\bence\projectderbiuj\data\trotting1012.db")
    query = "SELECT * FROM horse_races_aggregated"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)




df=pd.read_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv")

#getting dummies


#df.to_csv(r"C:\Users\bence\OneDrive\Project derbi\lasttestwithdummies.csv", index=False)

#reading data

df=df.drop(df[df['rank']==20].index)
df = df[df["id"]>146717]
df = df[df["id"]<161945]

print('Reading csv')
X=df[Xcolumns]

dummies=True

#Assigning X
def getdummies(df, X):
       
       ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
       encoded = ohe.fit_transform(df[categoricalcolumns])
       features = ohe.categories_
       print('Got dummies')
       X=pd.concat([X,encoded], axis=1)
       print('Assigned X')
       return X, features

#Assigning y

y=df['rank']
print('Assigned y')

#splitting data

if dummies==True:
       X, features = getdummies(df, X)
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
else:
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
       features=None

#preprocessing data

imp_mean = SimpleImputer(strategy='mean')
X_train.loc[:,sscolumns] = imp_mean.fit_transform(X_train.loc[:,sscolumns])
X_test.loc[:,sscolumns] = imp_mean.transform(X_test.loc[:,sscolumns])
print('Data imputed')
ss=preprocessing.StandardScaler()
X_train.loc[:,sscolumns]=ss.fit_transform(X_train.loc[:,sscolumns])
X_test.loc[:,sscolumns]=ss.transform(X_test.loc[:,sscolumns])
print('Data scaled')
le=LabelEncoder()
le.fit(Y_train)
Y_train=le.transform(Y_train)
Y_test=le.transform(Y_test)
print('y labeled')

#fitting model

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Hyperparameter tuning

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 50, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

models = {
    'LinearRegression': (LinearRegression(), {
        'fit_intercept': [True, False],  # Whether to calculate the intercept
        'positive': [True, False]  # Ensure coefficients are positive
    }),
    'Ridge': (Ridge(), {
        'alpha': [0.1, 1.0, 10.0],  # Regularization strength
        'fit_intercept': [True, False]
    }),
    'Lasso': (Lasso(), {
        'alpha': [0.1, 1.0, 10.0],  # Regularization strength
        'fit_intercept': [True, False]
    }),
    'ElasticNet': (ElasticNet(), {
        'alpha': [0.1, 1.0, 10.0],  # Regularization strength
        'l1_ratio': [0.1, 0.5, 0.9],  # Mix of L1 and L2 regularization
        'fit_intercept': [True, False]
    }),
    'RandomForestRegressor': (RandomForestRegressor(random_state=1), {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 50, 100, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'XGBoostRegressor': (XGBRegressor(random_state=1), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    'LightGBMRegressor': (LGBMRegressor(random_state=1), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    'HistGradientBoostingRegressor': (HistGradientBoostingRegressor(random_state=1), {
        'max_iter': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1]
    }),
    'SVR': (SVR(), {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }),
    'KNeighborsRegressor': (KNeighborsRegressor(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    'MLPRegressor': (MLPRegressor(random_state=1, max_iter=1000), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    })
}

best_model = None
best_r2_score = float('-inf')
best_model_name = None

# Store the R² score of each model

model_r2_scores = {}

# Iterate through models and perform grid search

for model_name, (model, param_distributions) in models.items():
    print(f"Training {model_name}...")
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, scoring='r2', cv=5, verbose=1, n_iter=50, random_state=1)
    grid_search.fit(X_train, Y_train)
    
    # Save the R² score of the current model

    model_r2_scores[model_name] = grid_search.best_score_

    # Check if this model is the best

    if grid_search.best_score_ > best_r2_score:
        best_r2_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_model_name = model_name
    print(f"Current best R² score: {best_r2_score}")

print(f"\nBest Model: {best_model_name} with R² score: {best_r2_score:.4f}")



print("Best Parameters:", grid_search.best_params_)

# Cross-validation

cv_scores = cross_val_score(best_model, X_train, Y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_scores.mean())

# Fit the best model

best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)


#Preprocessing data for export

X = df[Xcolumns]
imp_mean = SimpleImputer(strategy='mean')
X.loc[:,sscolumns] = imp_mean.fit_transform(X.loc[:,sscolumns])
print('Data imputed')
ss=preprocessing.StandardScaler()
X.loc[:,sscolumns]=ss.fit_transform(X.loc[:,sscolumns])
print('Data scaled')
le=LabelEncoder()
le.fit(y)
y=le.transform(y)
print('y labeled')
best_model.fit(X,y)

#exporting model and scalers

joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\regressors_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
print('model and scalers exported')

#plotting data

print(Y_test)
print(Y_pred)
print('R2score: ' , r2_score(Y_test, Y_pred))
print('Accuracy score: ', accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Feature importance

importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\feature_importance_oneyear.csv", index=False)

# Print the R² score of each model

print("\nModel R² Scores:")
for model_name, r2_score in model_r2_scores.items():
    print(f"{model_name}: {r2_score:.4f}")

