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
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from dotenv import load_dotenv
from pathlib import Path

project_root = Path(__file__).parent.parent 

import os

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



df=pd.read_csv(project_root / "data" / "querynewtop4.csv")

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
    'KNN': (KNeighborsClassifier(), {
         'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'LogisticRegression': (LogisticRegression(random_state=1, max_iter=1000), {
         'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2']}),
    'RandomForest': (RandomForestClassifier(random_state=1), {
         'n_estimators': [10, 100, 200], 'max_depth': [5, 7, 10, 15]}),
    'XGBoost': (XGBClassifier(random_state=1), {
         'n_estimators': [10, 100, 500], 'max_depth': [3, 6, 10]}),
    'LightGBM': (LGBMClassifier(random_state=1), {
         'n_estimators': [10, 100, 500], 'max_depth': [3, 6, 10]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=1), {
         'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}),
    'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=1), {
         'max_iter': [100, 200], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1]}),
    'MLP': (MLPClassifier(random_state=1, max_iter=1000), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}),
    'SVM': (SVC(random_state=1), {
        'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']})
}    

best_model = None
best_accuracy = 0
best_model_name = None

# Store the accuracy of each model
model_accuracies = {}

# Iterate through models and perform grid search
for model_name, (model, param_distributions) in models.items():
    print(f"Training {model_name}...")
    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, scoring='accuracy', cv=5, verbose=1, n_iter=5, random_state=1)
    grid_search.fit(X_train, Y_train)
    
    # Save the accuracy of the current model
    model_accuracies[model_name] = grid_search.best_score_
    # Check if this model is the best
    if grid_search.best_score_ > best_accuracy:
        best_accuracy = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_model_name = model_name
    print("current best accuracy: " , best_accuracy)

print(f"Best Model: {best_model_name} with accuracy: {best_accuracy}")



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
model_dir = Path(__file__).resolve().parent / "models"


joblib.dump(best_model, model_dir / "classifiers_oneyear.pkl")
joblib.dump(imp_mean, model_dir / "imputer_oneyear.pkl")
joblib.dump(ss, model_dir / "standardscaler_oneyear.pkl")
joblib.dump(features, model_dir / "features_oneyear.pkl")
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

# Print the accuracy of each model at the end
print("\nModel Accuracies:")
for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")

print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")