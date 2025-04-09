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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV


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
y = df['top4']



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

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    'class_weight': ['balanced', 'balanced_subsample', None]  # Class weights
}

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random combinations to try
    scoring='accuracy',  # Metric to optimize
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=1  # For reproducibility
)

# Fit RandomizedSearchCV
random_search.fit(X_train, Y_train)

# Get the best model and parameters
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Make predictions
Y_pred = best_model.predict(X_test)

# Export the best model and preprocessing objects
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modelrandomf_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
print('Best model and scalers exported')

# Evaluate the model
print('Accuracy score:', accuracy_score(Y_test, Y_pred))
print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred))

# Feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Export feature importance to CSV
feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\randomforestfeature_importance_randomsearch_oneyear.csv", index=False)

# Plot Actual vs Predicted
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()