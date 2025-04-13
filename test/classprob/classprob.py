import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # Required for HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Define columns
sscolumns = ['horse_prize_1y', 'horse_avg_km_time_6m',
             'horse_avg_km_time_12m', 'horse_min_km_time_6m',
             'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
             'horse_avg_km_time_improve_12m', 'horse_gals_1y',
             'horse_wins_1y', 'horse_podiums_1y', 'horse_fizetos_1y',
             'jockey_wins_1y', 'horse_wins_percent_1y',
             'horse_podiums_percent_1y', 'horse_fizetos_percent_1y', 'horse_age']

categoricalcolumns = ['race_length', 'num_horses']
labelcolumns = ['horse_id', 'stable_id', 'jockey_id']
Xcolumns = sscolumns

# Load data
df = pd.read_csv(r"C:\Users\bence\projectderbiuj\data\merged_output.csv")
#df=df[df['id']>146717] #2020
df=df[df['id']<161944] 
# Filter data
df = df[df['rank'] != 0]
df = df[df['rank'] < 13]

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
y = df['rank']

# One-Hot Encode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
features = ohe.categories_
X = pd.concat([X, encoded], axis=1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)

# Impute missing values
imp_mean = SimpleImputer(strategy='mean')
X_train.loc[:, sscolumns] = imp_mean.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = imp_mean.transform(X_test.loc[:, sscolumns])

# Scale numerical features
ss = StandardScaler()
X_train.loc[:, sscolumns] = ss.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = ss.transform(X_test.loc[:, sscolumns])

# Encode target variable (shift classes to start from 1)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train) + 1  # Shift classes to start from 1
Y_test = le.transform(Y_test) + 1        # Shift classes to start from 1

# Define models
models = {
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=1),
}

# Track the best model
best_model_name = None
best_model = None
best_log_loss = float('inf')

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    
    # Use predict_proba for probability-based evaluation
    if hasattr(model, "predict_proba"):
        Y_pred_proba = model.predict_proba(X_test)
        logloss = log_loss(Y_test - 1, Y_pred_proba)  # Adjust Y_test back to 0-based indexing for log_loss
        print(f"{name} Log Loss: {logloss}")
        
        # Update the best model if this model performs better
        if logloss < best_log_loss:
            best_log_loss = logloss
            best_model_name = name
            best_model = model
    else:
        print(f"{name} does not support probability predictions.")

# Print the best model
print(f"\nBest Model: {best_model_name} with Log Loss: {best_log_loss}")

# Print predicted probabilities compared to real values
if hasattr(best_model, "predict_proba"):
    print("\nPredicted Probabilities vs Real Values:")
    probabilities = best_model.predict_proba(X_test)
    for i in range(len(Y_test)):
        print(f"Actual: {Y_test[i]}, Predicted Probabilities: {probabilities[i]}")
else:
    print(f"The best model ({best_model_name}) does not support probability predictions.")

# Print predicted probabilities, the class with the highest probability, and the actual values
if hasattr(best_model, "predict_proba"):
    print("\nPredicted Probabilities, Predicted Class, and Actual Values:")
    probabilities = best_model.predict_proba(X_test)
    predicted_classes = np.argmax(probabilities, axis=1) + 1  # Shift predicted classes to start from 1
    for i in range(len(Y_test)):
        print(f"Actual: {Y_test[i]}, Predicted Probabilities: {probabilities[i]}, Predicted Class: {predicted_classes[i]}")
else:
    print(f"The best model ({best_model_name}) does not support probability predictions.")

# Print the log loss of the best model at the end
print(f"\nFinal Log Loss of the Best Model ({best_model_name}): {best_log_loss}")

best_model.fit(X,y)

# Define the parameter grid for HistGradientBoostingClassifier
param_grid = {
    "learning_rate": [0.01],
    "max_iter": [200],
    "max_depth": [None],
    "min_samples_leaf": [30],
    "l2_regularization": [10.0]
}

# Initialize the HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(class_weight="balanced", random_state=1)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,  # Number of random combinations to try
    scoring="neg_log_loss",  # Use log loss as the scoring metric for classification
    cv=3,  # 3-fold cross-validation
    verbose=1,
    random_state=1,
    n_jobs=-1  # Use all available cores
)

# Fit the randomized search on the training data
print("Performing Randomized Search...")
random_search.fit(X_train, Y_train)

# Get the best model and parameters
best_model = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

# Evaluate the best model on the test set
if hasattr(best_model, "predict_proba"):
    Y_pred_proba = best_model.predict_proba(X_test)
    logloss = log_loss(Y_test, Y_pred_proba)
    print(f"Log Loss on Test Set: {logloss}")

y_pred=best_model.predict(X_test) # Predict classes on the test set
# Print the distribution of predicted classes
from collections import Counter
print("\nDistribution of Predicted Classes:")
print(Counter(y_pred))
# Preprocess the entire dataset
print("\nPreprocessing the entire dataset...")
X.loc[:, sscolumns] = imp_mean.fit_transform(X.loc[:, sscolumns])  # Impute missing values
X.loc[:, sscolumns] = ss.fit_transform(X.loc[:, sscolumns])  # Scale numerical features

# Encode the target variable
y = le.fit_transform(y)

# Train the best model on the full dataset
print("\nTraining the best model on the full dataset...")
best_model.fit(X, y + 1)  # Shift y to start from 1

# Export preprocessing objects
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(ohe, r"C:\Users\bence\projectderbiuj\models\onehotencoder_oneyear.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
print('Preprocessing objects exported')

# Export the best model
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\classprob_model.pkl")
print(f"Best model exported")
