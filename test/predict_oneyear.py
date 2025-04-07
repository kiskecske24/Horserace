import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.compose import ColumnTransformer


model_fit=joblib.load(r'C:\Users\bence\projectderbiuj\models\modelcomplex_oneyear.pkl')
imp_mean=joblib.load(r'C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl')
ss=joblib.load(r'C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl')
features=joblib.load(r'C:\Users\bence\projectderbiuj\models\features_oneyear.pkl')

# time-based and percentage
minmax_columns = [

  'horse_wins_percent_1y', 'horse_wins_percent_2y', 'horse_wins_percent_3y',

  'horse_podiums_percent_1y', 'horse_podiums_percent_2y', 'horse_podiums_percent_3y',
  'horse_fizetos_percent_1y', 'horse_fizetos_percent_2y', 'horse_fizetos_percent_3y',
  'horse_min_km_time_improve_12m', 
  'horse_min_km_time_improve_2y', 
  'horse_avg_km_time_improve_12m',  
  'horse_avg_km_time_improve_2y',
  'horse_min_km_time_improve',
  'horse_avg_km_time_improve',
  'number',
  
]



# count-based and time based?

standard_columns = [
  'horse_min_km_time_6m', 'horse_min_km_time_12m', 
  'horse_min_km_time_1y',
  'horse_min_km_time_2y', 
'horse_min_km_time_3y',
  'horse_avg_km_time_6m', 'horse_avg_km_time_12m',
  'horse_avg_km_time_1y',
  'horse_avg_km_time_2y', 
'horse_avg_km_time_3y',
  'horse_prize_1y','horse_prize_2y','horse_prize_3y',

  'horse_gals_1y','horse_gals_2y','horse_gals_3y',

  'horse_wins_1y','horse_wins_2y','horse_wins_3y',

  'horse_podiums_1y','horse_podiums_2y','horse_podiums_3y',

  'horse_fizetos_1y','horse_fizetos_2y','horse_fizetos_3y',
 
  'jockey_wins_1y','jockey_wins_2y','jockey_wins_5y',
  'horse_age',
  'dividend',
]

categoricalcolumns=['number','race_length','jockey_id','stable_id','trainer_id']


base_query = """
SELECT
    race_id,
    horse_id,
    jockey_id,
    trainer_id,
    rank,
    km_time,
    dividend,
    prize,
    race_length, race_date, stable_id,
    dividend as dividend_real,
    number as number_real

"""

dynamic_columns = ', '.join(standard_columns + minmax_columns)

MAIN_QUERY = f"""
{base_query}, {dynamic_columns}
FROM horse_races_aggregated hr
"""  

# ez sqlite romai szamo sorrendezeshez kell
def roman_to_integer(roman):
    debug_file = "/tmp/roman_debug.log"
    
    try:
        # Open the debug file in append mode
        with open(debug_file, "a") as f:
            if roman is None or roman == "":
                f.write(f"Received empty or null value: {roman}\n")
                return 0

            roman = roman.rstrip('.')

            f.write(f"Processing Roman numeral: {roman}\n")

            roman = roman.upper()  # Convert to uppercase
            roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

            integer = 0
            prev_value = 0

            # Process each character in reverse
            for char in reversed(roman):
                if char not in roman_dict:
                    error_message = f"Invalid Roman numeral character: {char}\n"
                    f.write(error_message)
                    raise ValueError(error_message)

                value = roman_dict[char]
                if value < prev_value:
                    integer -= value
                else:
                    integer += value

                prev_value = value

            f.write(f"Converted {roman} to {integer}\n")
            return integer

    except Exception as e:
        with open(debug_file, "a") as f:
            f.write(f"Error processing Roman numeral '{roman}': {str(e)}\n")
        raise





columns=['horse_id','dividend','jockey_id','km_time','rank','prize',
         'race_id','race_length','race_date','stable_id','trainer_id','number',
         'horse_prize_1y','horse_prize_2y','horse_prize_3y','horse_avg_km_time_1y','horse_avg_km_time_2y',
         'horse_avg_km_time_3y','horse_avg_km_time_6m','horse_avg_km_time_12m','horse_min_km_time_6m','horse_min_km_time_12m',
         'horse_min_km_time_1y'	,'horse_min_km_time_2y'	,'horse_min_km_time_3y','horse_min_km_time_improve_12m'	,
         'horse_min_km_time_improve_2y'	,'horse_avg_km_time_improve_12m','horse_avg_km_time_improve_2y','horse_gals_1y',
         'horse_gals_2y','horse_gals_3y','horse_wins_1y','horse_wins_2y','horse_wins_3y','horse_podiums_1y','horse_podiums_2y',
         'horse_podiums_3y','horse_fizetos_1y','horse_fizetos_2y','horse_fizetos_3y','jockey_wins_1y','jockey_wins_2y','jockey_wins_5y',
         'horse_wins_percent_1y','horse_wins_percent_2y','horse_wins_percent_3y','horse_podiums_percent_1y','horse_podiums_percent_2y',
         'horse_podiums_percent_3y','horse_fizetos_percent_1y','horse_fizetos_percent_2y','horse_fizetos_percent_3y']

sscolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns=['number','race_length','horse_id','stable_id','jockey_id' ]

Xcolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

conn = sqlite3.connect(r'C:\Users\bence\projectderbiuj\data\trotting1012.db')
conn.create_function("roman_to_integer", 1, roman_to_integer)
cur = conn.cursor()
cur.execute("SELECT date,id,daily from races where date='2024-10-12' and id=18284")
races = cur.fetchall()
print(races)

for race in races:
  print(race)
  query = MAIN_QUERY + " where race_id=" + str(race[1]) + " order by number"
  pd.set_option('display.max_columns', None)
  df = pd.read_sql(query, conn)

df=pd.read_csv(r'C:\Users\bence\projectderbiuj\data\querymewtop4.csv')


def getresults():
    race_id=18284
    ohe = OneHotEncoder(categories=features, handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

    filtered_df = df[df['race_id'] == race_id]
    X=filtered_df[Xcolumns]
    encoded = ohe.fit_transform(df[categoricalcolumns])
    X=pd.concat([X,encoded], axis=1)
    X.loc[:,sscolumns] = imp_mean.transform(X.loc[:,sscolumns])
    X.loc[:,sscolumns]=ss.fit_transform(X.loc[:,sscolumns])
    originaldatabase=pd.read_csv(r'C:\Users\bence\projectderbiuj\data\querynewtop4.csv')
    fornumberdatabase = originaldatabase[originaldatabase['race_id'] == race_id]
    Y_pred=model_fit.predict(X)
    numbers=fornumberdatabase.number.tolist()
    actual=fornumberdatabase['rank'].tolist()
    for x in range(len(Y_pred)):
        print('Number: ', numbers[x], 'Prediction: ', Y_pred[x], 'Actual: ', actual[x])
getresults()
    



