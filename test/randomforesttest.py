import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Define columns
sscolumns = ['horse_prize_1y', 'horse_avg_km_time_6m',
             'horse_avg_km_time_12m', 'horse_min_km_time_6m',
             'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
             'horse_avg_km_time_improve_12m', 'horse_gals_1y',
             'horse_wins_1y', 'horse_podiums_1y', 'horse_fizetos_1y',
             'jockey_wins_1y', 'horse_wins_percent_1y',
             'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns = ['race_length','horse_age']
labelcolumns=['horse_id', 'stable_id', 'jockey_id']


Xcolumns = sscolumns

from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv(r"C:\Users\bence\projectderbiuj\data\merged_output.csv")

# Filter data


df = df[df['rank'] != 0]

#df = df[(df["id"] > 146717)]

# Handle missing values in competitor columns (if any)
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'].fillna(-1, inplace=True)  # Fill missing values with -1

# Apply Label Encoding to competitor columns
le = LabelEncoder()
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'] = le.fit_transform(df[f'competitor_{i}'].astype(str))
    
# Debug: Print the first few rows to verify encoding
print(df[[f'competitor_{i}' for i in range(1, 14)]].head())
for x in range(0,2):
    df[labelcolumns[x]]=le.fit_transform(df[labelcolumns[x]].astype(str))
# Assign X and y
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]

y = df['top4']


dummies=False
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

y=df['top4']
print('Assigned y')

#splitting data
if dummies==True:
       X, features = getdummies(df, X)
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
else:
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
       features=None
print('Data splitted')

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

best_model = RandomForestClassifier(random_state=1, class_weight='balanced', n_estimators= 100, min_samples_split= 5,
                                     min_samples_leaf=1, max_features= 'sqrt', max_depth= 10)

best_model.fit(X_train, Y_train)
# Evaluate the best model on the test set
Y_pred = best_model.predict(X_test)
print('Test Accuracy:', accuracy_score(Y_test, Y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred))

# Export model and preprocessing objects
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modelrandomf_oneyear_randomsearch.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
print('Model and scalers exported')

# Evaluate model
print('R2 Score:', r2_score(Y_test, Y_pred))
print('Accuracy Score:', accuracy_score(Y_test, Y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred))

# Feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Save feature importance
feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\randomforestfeature_importance_oneyear_randomsearch.csv", index=False)

# Train with whole dataset
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]
y = df['top4']
y = le.fit_transform(y)
best_model.fit(X, y)

print('Model trained with whole dataset')

# Export model and preprocessing objects
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modelrandomf_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
print('Model and scalers exported')

# Evaluate model
print('R2 Score:', r2_score(Y_test, Y_pred))
print('Accuracy Score:', accuracy_score(Y_test, Y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred))

# Feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Save feature importance
feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\randomforestfeature_importance_oneyear.csv", index=False)

#Train with whole dataset


# Handle missing values in competitor columns (if any)
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'].fillna(-1, inplace=True)  # Fill missing values with -1

# Apply Label Encoding to competitor columns
le = LabelEncoder()
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'] = le.fit_transform(df[f'competitor_{i}'].astype(str))

# Debug: Print the first few rows to verify encoding
print(df[[f'competitor_{i}' for i in range(1, 14)]].head())

# Assign X and y
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)]+labelcolumns]
y = df['top4']
y=le.fit_transform(y)
best_model = RandomForestClassifier(random_state=1, class_weight='balanced')
best_model.fit(X, y)

print('Model trained with whole dataset')