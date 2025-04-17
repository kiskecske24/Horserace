#!/usr/bin/env python3
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import sqlite3

from pathlib import Path
project_root = model_dir = Path(__file__).resolve().parent.parent.parent
models_dir = project_root / "models"

# Load pre-trained model and preprocessing objects
best_model = joblib.load( models_dir  / "classprob_model.pkl")
imp_mean = joblib.load(models_dir / "imputer_oneyear.pkl")
ss = joblib.load(models_dir / "standardscaler_oneyear.pkl")
features=joblib.load(models_dir / "features_oneyear.pkl")
ohe=joblib.load(models_dir / "onehotencoder_oneyear.pkl")

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
df= pd.read_csv(project_root / "data" / "merged_output.csv")

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
    Predict and display the top 4 horses with the highest probability of being in the top 4,
    along with the predicted classes of all horses.
    """
    # Preprocess the data
    X, filtered_df = preprocess_data(df, race_id)

    # Make predictions (probabilities and classes)
    Y_proba = best_model.predict_proba(X)  # Predicted probabilities
    Y_pred = best_model.predict(X)        # Predicted classes

    # Calculate probabilities for being in the top 4
    prob_top4 = Y_proba[:, 0:4].sum(axis=1)  # Sum of probabilities for classes 1, 2, 3, and 4

    # Get additional information for display
    numbers = filtered_df['number'].tolist()

    # Combine results into a list of tuples
    results = list(zip(numbers, prob_top4, Y_pred))

    # Sort by probabilities for top 4
    results.sort(key=lambda x: x[1], reverse=True)
    top_4_top4 = results[:4]  # Top 4 horses for top 4
    print(top_4_top4)

    # Print the top 4 horses with the highest probability of being in the top 4
    print("==================================================================")
    print("\nTop 4 Horses with the Highest Probability of Being in the Top 4:")
    for i, (number, prob_top4, _) in enumerate(top_4_top4, start=1):
        print(f"{i}. Horse Number {number}, Probability Top 4: {prob_top4:.4f}")

    # Print the predicted classes of all horses
    print("\nPredicted Classes of All Horses:")
    for number, _, predicted_class in results:
        print(f"Horse Number {number}, Predicted Class: {predicted_class}")
    return top_4_top4

# Call the function with a specific race_id

import sqlite3

conn = sqlite3.connect(project_root / "data" / 'trotting1012.db')  # replace with your actual DB path
cursor = conn.cursor()

query = """
SELECT id
FROM races
WHERE date >= '2023-01-01' and date< '2024-01-01'
  AND daily NOT IN ('P', 'Q');
"""
#WHERE date >= '2024-01-01' and date< '2024-10-12'

# nincsen rajuk adat
# 18275 - 3 jo
# 18281 - 3-bol kettot tippel meg

cursor.execute(query)
race_ids = [row[0] for row in cursor.fetchall()]
spend = 0


conn.row_factory = sqlite3.Row  # ✅ Enables dict-like row access
cursor = conn.cursor()


def get_race(race_id):
  query = " SELECT * from races where id=?"
  cursor.execute(query, (race_id,))
  res = cursor.fetchone()
  return res

def num_horses(race_id):
  query = "SELECT COUNT(*) as count from horse_races where race_id=?"
  cursor.execute(query,(race_id,))
  res = cursor.fetchone()
  return res["count"]

for race_id in race_ids:
  race = get_race(race_id)
  print("RACE_ID: " , race["id"] , "Date: ", race["date"])
  predicted_top = getresults(race_id=race_id)
#   print("PREDICTED")
#   print(predicted_top)

  query = """
  SELECT number, rank
  FROM horse_races
  WHERE race_id = ?
    AND rank IN ('1', '2', '3');
  """

  cursor.execute(query, (race_id,))
  actual_top = cursor.fetchall()
#   print("ACTUAL")
#   print(actual_top)
#   conn.close()
  actual_top_numbers = [int(row[0]) for row in actual_top]
  predicted_numbers = [row[0] for row in predicted_top]
  correct_predictions = [num for num in predicted_numbers if num in actual_top_numbers]
  print("Predicted top:", predicted_numbers)
  print("Actual top 3 finishers:", actual_top_numbers)
  print("Matched predictions:", correct_predictions)

  query = """
  SELECT dividend_trifecta
  FROM races
  WHERE id=?
  """
  cursor.execute(query,(race_id,))
  result = cursor.fetchone()
  if result is not None and result[0] is not None:
    dividend = result[0]
  else:
    print("NINCSEN DIVIDEND")
  print("DIVIDEND: ", dividend)
  bet = 0

  if dividend > 50:
    spend = spend -3600
    bet = 1

  if len(correct_predictions) == 3 and bet == 1: 
    print("NYERTUNK")
    spend = spend + 200*dividend
    num = num_horses(race_id)
    print("Num horses: ", num)

  print("BALANCE: ", spend)
#   break

