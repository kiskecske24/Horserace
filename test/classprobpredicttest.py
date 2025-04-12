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
ohe = joblib.load(r'C:\Users\bence\projectderbiuj\models\onehotencoder_oneyear.pkl')  # Load pre-fitted OneHotEncoder

# Define columns
Xcolumns = [
    'horse_prize_1y', 'horse_avg_km_time_6m', 'horse_avg_km_time_12m',
    'horse_min_km_time_6m', 'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
    'horse_avg_km_time_improve_12m', 'horse_gals_1y', 'horse_wins_1y',
    'horse_podiums_1y', 'horse_fizetos_1y', 'jockey_wins_1y',
    'horse_wins_percent_1y', 'horse_podiums_percent_1y', 'horse_fizetos_percent_1y'
]
sscolumns = Xcolumns
labelcolumns = ['horse_id', 'stable_id', 'jockey_id']
categoricalcolumns = ['race_length', 'horse_age']

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
    encoded = ohe.transform(filtered_df[categoricalcolumns])  # Transform using pre-fitted encoder

    # Combine encoded categorical features with the rest of the features
    X = pd.concat([filtered_df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns], encoded], axis=1)

    # Impute and scale numerical features
    X.loc[:, sscolumns] = imp_mean.transform(X.loc[:, sscolumns])
    X.loc[:, sscolumns] = ss.transform(X.loc[:, sscolumns])

    return X, filtered_df

def getresults(race_id):
    """
    Predict and display results for a specific race_id.
    """
    # Load the dataset
    df = pd.read_csv(r'C:\Users\bence\projectderbiuj\data\merged_output.csv')

    # Preprocess the data
    X, filtered_df = preprocess_data(df, race_id)

    # Make predictions
    Y_pred_proba = best_model.predict_proba(X)  # Predicted probabilities

    # Get additional information for display
    numbers = filtered_df['number'].tolist()
    actual = filtered_df['rank'].tolist()

    # Calculate the sum of probabilities for being in the 1-4 range
    prob_1_to_4 = Y_pred_proba[:, :4].sum(axis=1)  # Sum probabilities for classes 1, 2, 3, and 4

    # Combine results into a DataFrame
    results_df = pd.DataFrame({
        'Number': numbers,
        'Actual Rank': actual,
        'Prob 1-4': prob_1_to_4
    })

    # Sort the DataFrame by the sum of probabilities for being in the 1-4 range (descending order)
    results_df = results_df.sort_values(by='Prob 1-4', ascending=False)

    # Display results
    print("\nResults (Ordered by Probability of Being in 1-4 Range):")
    for _, row in results_df.iterrows():
        print(f"Number: {row['Number']}, Prob 1-4: {row['Prob 1-4']:.4f}, Actual Rank: {row['Actual Rank']}")

# Call the function with a specific race_id
getresults(race_id=18284)