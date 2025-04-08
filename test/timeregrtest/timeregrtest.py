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
from sklearn.model_selection import RandomizedSearchCV
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
df = df[df['time'] != 0]


#df=df[df['id']>146717]

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

# Drop rows where the target variable 'km_time' (y) is NaN
df = df.dropna(subset=['time'])

# Assign X and y
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]
y = df['time']

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



# Define models
# Define regression models
# Define regression models
models = {
    "LightGBM": LGBMRegressor(random_state=1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=1),
    "XGBoost": XGBRegressor(random_state=1),
    "KNeighbors": KNeighborsRegressor(),
    "MLP": MLPRegressor(random_state=1, max_iter=1000),
}

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    "LightGBM": {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [3, 5, 10, -1],  # -1 means no limit in LightGBM
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 50, 100],
        'min_child_samples': [10, 20, 30],
    },
    "HistGradientBoosting": {
        'max_iter': [100, 200, 500],
        'max_depth': [3, 5, 10, None],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0.0, 0.1, 1.0],
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
    },
    "KNeighbors": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    },
}

# Initialize variables to store the best model
best_model_name = None
best_model = None
best_r2_score = float('-inf')

# Perform RandomizedSearchCV for each model
for name, model in models.items():
    print(f"\nPerforming Randomized Search for {name}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[name],
        scoring='r2',
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1,  # Use all available CPU cores
        n_iter=10,  # Number of random combinations to try
        random_state=1  # For reproducibility
    )
    random_search.fit(X_train, Y_train)
    
    # Get the best model and evaluate it
    best_model_for_current = random_search.best_estimator_
    Y_pred = best_model_for_current.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    
    print(f"{name} Best Parameters: {random_search.best_params_}")
    print(f"{name} R2 Score: {r2}")
    
    # Update the best model if this model performs better
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_name = name
        best_model = best_model_for_current

# Print the best model
print("\nBest Model:")
print(f"Model Name: {best_model_name}")
print(f"R2 Score: {best_r2_score}")

# Export the best model
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\best_regression_model_randomsearch.pkl")
print(f"Best model ({best_model_name}) exported")

# Scatter plot of actual vs predicted values for the best model
Y_pred = best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', label='Predicted vs Actual')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title(f"Best Model: {best_model_name} (R2: {best_r2_score:.4f})")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)
plt.show()