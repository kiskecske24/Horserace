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
from lightgbm import LGBMClassifier
import joblib

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
df = pd.read_csv(r"C:\Users\bence\projectderbiuj\data\class_by_3merged.csv")

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
y = df['class_by_3']

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

# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 30, 50],
    'reg_alpha': [0.0, 0.1, 1.0, 10.0],
    'reg_lambda': [0.0, 0.1, 1.0, 10.0]
}

# Initialize the LightGBMClassifier
lgbm_model = LGBMClassifier(random_state=1)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings to sample
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=1
)

# Fit RandomizedSearchCV
print("Starting Randomized Search...")
random_search.fit(X_train, Y_train)

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_cross_val_score = random_search.best_score_

print("\nBest Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_cross_val_score)

# Evaluate the best model on the test set
Y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred)
print("\nTest Accuracy:", test_accuracy)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Export model and preprocessing objects
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modellgbm_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(ohe, r"C:\Users\bence\projectderbiuj\models\onehotencoder_oneyear.pkl")
print('Model and scalers exported')

# Evaluate model
print("\nR2 Score:", r2_score(Y_test, Y_pred))
print("Accuracy Score:", test_accuracy)
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Train with the whole dataset
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]
X = pd.concat([X, ohe.transform(df[categoricalcolumns])], axis=1)
X.loc[:, sscolumns] = imp_mean.transform(X.loc[:, sscolumns])
X.loc[:, sscolumns] = ss.transform(X.loc[:, sscolumns])
y = le.fit_transform(df['class_by_3'])

best_model.fit(X, y)
print("\nModel trained with the whole dataset")

# Export final model
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modellgbm_oneyear_final.pkl")
print("\nFinal model exported")

# Print the best model and its parameters
print("\nBest Model Summary:")
print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_cross_val_score}")
print(f"Test Accuracy: {test_accuracy}")