import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load pre-trained model and preprocessing objects
best_model = joblib.load(r'C:\Users\bence\projectderbiuj\models\classprob_model.pkl')
imp_mean = joblib.load(r'C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl')
ss = joblib.load(r'C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl')
features=joblib.load(r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
ohe=joblib.load(r"C:\Users\bence\projectderbiuj\models\onehotencoder_oneyear.pkl")

# Define columns
Xcolumns = [
    'horse_prize_1y', 'horse_avg_km_time_6m', 'horse_avg_km_time_12m',
    'horse_min_km_time_6m', 'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
    'horse_avg_km_time_improve_12m', 'horse_gals_1y', 'horse_wins_1y',
    'horse_podiums_1y', 'horse_fizetos_1y', 'jockey_wins_1y',
    'horse_wins_percent_1y', 'horse_podiums_percent_1y', 'horse_fizetos_percent_1y', 'horse_age'
]
sscolumns = Xcolumns
labelcolumns = ['horse_id', 'stable_id', 'jockey_id']
categoricalcolumns = ['race_length', 'num_horses']
df= pd.read_csv(r"C:\Users\bence\projectderbiuj\data\merged_output.csv")

def preprocess_data(df, race_id):
    """
    Preprocess the data for the given race_id.
    """
    # Filter data for the specific race
    filtered_df = df[df['race_id'] == race_id]

    # Handle missing values in competitor columns
    for i in range(1, 14):  # For competitor_1 to competitor_14
        filtered_df[f'competitor_{i}'].fillna(-1, inplace=True)

    # Apply Label Encoding to competitor columns
    le = LabelEncoder()
    for i in range(1, 14):  # For competitor_1 to competitor_14
        filtered_df[f'competitor_{i}'] = le.fit_transform(filtered_df[f'competitor_{i}'].astype(str))

    # One-Hot Encode categorical columns using the pre-fitted encoder
    ohe = OneHotEncoder(categories=features, handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    encoded = ohe.fit_transform(filtered_df[categoricalcolumns])  # Transform using pre-fitted encoder

    # Combine encoded categorical features with the rest of the features
    X = pd.concat([filtered_df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns], encoded], axis=1)

    # Impute and scale numerical features
    X.loc[:, sscolumns] = imp_mean.transform(X.loc[:, sscolumns])
    X.loc[:, sscolumns] = ss.transform(X.loc[:, sscolumns])

    return X, filtered_df

def getresults(race_id):
    """
    Predict and display results for a specific race_id in the specified format:
    I. Top 2 horses for 1st position.
    II. Top 4 horses for 2-4 range.
    III. Combined list of horses from I and II.
    """
    # Preprocess the data
    X, filtered_df = preprocess_data(df, race_id)

    # Make predictions (probabilities)
    Y_proba = best_model.predict_proba(X)  # Predicted probabilities

    # Calculate probabilities for being in the 1st position
    prob_1st = Y_proba[:, 0:3].sum(axis=1)  # Class 1 probability (1st place)

    # Calculate probabilities for being in the 2-4 range
    prob_2_to_4 = Y_proba[:, 1:4].sum(axis=1)  # Sum of probabilities for classes 2, 3, and 4

    # Calculate probabilities for being in the top 4
    prob_top4 = Y_proba[:, 0:4].sum(axis=1)  # Sum of probabilities for classes 1, 2, 3, and 4
    # Get additional information for display
    numbers = filtered_df['number'].tolist()

    # Combine results into a list of tuples
    results = list(zip(numbers, prob_1st, prob_2_to_4))

    # Sort by probabilities for 1st place
    results.sort(key=lambda x: x[1], reverse=True)
    top_2_1st = results[:2]  # Top 2 horses for 1st place

    # Sort by probabilities for 2-4 range
    results.sort(key=lambda x: x[2], reverse=True)
    top_4_2_to_4 = results[:3]  # Top 2 horses for 2-4 range

    # Combine the results for I, II, and III
    combined_horses = list({horse[0] for horse in top_2_1st + top_4_2_to_4})  # Unique horse numbers

    # Print results in the specified format
    print("\nI.")
    for i, (number, prob_1st, _) in enumerate(top_2_1st, start=1):
        print(f"{i}. Horse Number {number}, Probability 1st: {prob_1st:.4f}")

    print("\nII.")
    for i, (number, prob_1st, _) in enumerate(top_2_1st, start=1):
        print(f"{i}. Horse Number {number}, Probability 1st: {prob_1st:.4f}")
    for i, (number, _, prob_2_to_4) in enumerate(top_4_2_to_4, start=1):
        print(f"{i}. Horse Number {number}, Probability 2-4: {prob_2_to_4:.4f}")

    print("\nIII.")
    for i, (number, prob_1st, _) in enumerate(top_2_1st, start=1):
        print(f"{i}. Horse Number {number}, Probability 1st: {prob_1st:.4f}")
    for i, (number, _, prob_2_to_4) in enumerate(top_4_2_to_4, start=1):
        print(f"{i}. Horse Number {number}, Probability 2-4: {prob_2_to_4:.4f}")

    
# Call the function with a specific race_id
getresults(race_id=18284)