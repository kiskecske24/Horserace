import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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

#df =df[df['id'] >98735] #2010


df = df[df["id"]>146717] #2020

#df = df[df["id"]>153503] #2022

df=df[df['rank']!=0]
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

ss=StandardScaler()
df.loc[:, 'km_time_new'] = ss.fit_transform(df.loc[:, 'km_time_new'].values.reshape(-1, 1))
y = df['km_time_new']



# One-Hot Encode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
features = ohe.categories_
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
# Add CatBoostRegressor to the models dictionary
models = {
    "LightGBM": LGBMRegressor(random_state=1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=1),
    "XGBoost": XGBRegressor(random_state=1),
}

# Add CatBoostRegressor hyperparameters to the grid
param_distributions = {
    "LightGBM": {
        'n_estimators': [500],
        'max_depth': [-1],  # -1 means no limit in LightGBM
        'learning_rate': [0.05],
        'num_leaves': [50],
        'min_child_samples': [10],
    },
    "HistGradientBoosting": {
        'max_iter': [500],
        'max_depth': [10],
        'learning_rate': [0.1],
        'min_samples_leaf': [10],
        'l2_regularization': [0.1],
    },
    "XGBoost": {
        'n_estimators': [200],
        'max_depth': [10],
        'learning_rate': [0.05],
        'subsample': [0.8],
    },

}

# Initialize variables to track the best model
best_r2_score = float('-inf')  # Set to negative infinity to ensure any R2 score will be higher
best_model_name = None
best_model = None

# Perform GridSearchCV for each model
for name, model in models.items():
    print(f"\nPerforming Grid Search for {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_distributions[name],  # Use param_distributions as param_grid
        scoring='r2',
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1  # Use all available CPU cores
    )
    grid_search.fit(X_train, Y_train)
    
    # Get the best model and evaluate it
    best_model_for_current = grid_search.best_estimator_
    Y_pred = best_model_for_current.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    
    print(f"{name} Best Parameters: {grid_search.best_params_}")
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
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\best_regression_model_gridsearch.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\scaler.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer.pkl")
print(f"Best model ({best_model_name}) exported")

# Scatter plot of actual vs predicted values for the best model
Y_pred = best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', s=10, label='Predicted vs Actual')  # Adjusted the size with 's=10'
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title(f"Best Model: {best_model_name} (R2: {best_r2_score:.4f})")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

# Print most important features (if supported by the model)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Print top 10 features
    print("\nMost Important Features:")
    print(feature_importance_df.head(10))  # Print top 10 features
    
    # Export feature importance to a CSV file
    feature_importance_csv_path = r"C:\Users\bence\projectderbiuj\data\feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_csv_path, index=False)
    print(f"\nFeature importance exported to {feature_importance_csv_path}")
else:
    print("\nThe best model does not support feature importance.")