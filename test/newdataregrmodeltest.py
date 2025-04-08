import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Define columns
sscolumns = ['horse_prize_1y', 'horse_avg_km_time_6m',
             'horse_avg_km_time_12m', 'horse_min_km_time_6m',
             'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
             'horse_avg_km_time_improve_12m', 'horse_gals_1y',
             'horse_wins_1y', 'horse_podiums_1y', 'horse_fizetos_1y',
             'jockey_wins_1y', 'horse_wins_percent_1y',
             'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns = ['race_length', 'horse_age']
labelcolumns = ['horse_id', 'stable_id', 'jockey_id']
Xcolumns = sscolumns

# Load data
df = pd.read_csv(r"C:\Users\bence\projectderbiuj\data\merged_output.csv")

# Filter data
df = df[df['rank'] != 0]

# Handle missing values in competitor columns (if any)
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'].fillna(-1, inplace=True)  # Fill missing values with -1

# Apply Label Encoding to competitor columns
le = LabelEncoder()
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'] = le.fit_transform(df[f'competitor_{i}'].astype(str))

# Apply Label Encoding to label columns
for col in labelcolumns:
    df[col] = le.fit_transform(df[col].astype(str))

# Assign X and y
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]
y = df['top4']

# One-Hot Encode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
X = pd.concat([X, encoded], axis=1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

# Impute missing values
imp_mean = SimpleImputer(strategy='mean')
X_train.loc[:, sscolumns] = imp_mean.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = imp_mean.transform(X_test.loc[:, sscolumns])

# Scale numerical features
ss = StandardScaler()
X_train.loc[:, sscolumns] = ss.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = ss.transform(X_test.loc[:, sscolumns])

# Encode target variable
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Define models
# Define regression models
models = {
    "RandomForest": RandomForestRegressor(random_state=1),
    "XGBoost": XGBRegressor(random_state=1),
    "LightGBM": LGBMRegressor(random_state=1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=1),
    "ExtraTrees": ExtraTreesRegressor(random_state=1),
    "MLPRegressor": MLPRegressor(max_iter=500, random_state=1),
    "SVR": SVR(),
    "KNeighbors": KNeighborsRegressor()
}

# Train and evaluate each regression model
best_model_name = None
best_model = None
best_r2_score = float('-inf')

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(Y_test, Y_pred)
    print(f"{name} R2 Score: {r2}")
    print(f"{name} Mean Absolute Error: {np.mean(np.abs(Y_test - Y_pred))}")
    print(f"{name} Mean Squared Error: {np.mean((Y_test - Y_pred) ** 2)}")
    
    # Check for feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        print(f"{name} Feature Importances:\n{feature_importance_df.sort_values(by='Importance', ascending=False)}")
    
    # Update the best model if this model performs better
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_name = name
        best_model = model

# Print the best model
print("\nBest Model:")
print(f"Model Name: {best_model_name}")
print(f"R2 Score: {best_r2_score}")

# Export preprocessing objects
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(ohe, r"C:\Users\bence\projectderbiuj\models\onehotencoder_oneyear.pkl")
print('Preprocessing objects exported')

# Export the best model
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\best_regression_model.pkl")
print(f"Best model ({best_model_name}) exported")