import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


sscolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y',
       'jockey_wins_1y','horse_wins_percent_1y',
       'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns=['number','race_length','horse_id','stable_id','jockey_id']

Xcolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y',
       'jockey_wins_1y','horse_wins_percent_1y',
       'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

#getting data
def getdata():
    conn = sqlite3.connect("trottingnew1012.db")
    query = "SELECT * FROM horse_races_aggregated"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(r"C:\Users\bence\OneDrive\Project derbi\querynew.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)



df=pd.read_csv('querynewtop5.csv')

df=df.drop(df[df['rank']==20].index)
df = df[df["id"]>146717]
df = df[df["id"]<161945]

print('Reading csv')
#Assigning X
X=df[Xcolumns]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
features = ohe.categories_
print('Got dummies')
X=pd.concat([X,encoded], axis=1)
print('Assigned X')
#Assigning y
y=df['rank']
print('Assigned y')

#preprocessing data
imp_mean = SimpleImputer(strategy='mean')
X.loc[:,sscolumns] = imp_mean.fit_transform(X.loc[:,sscolumns])

print('Data imputed')
ss=preprocessing.StandardScaler()
X.loc[:,sscolumns]=ss.fit_transform(X.loc[:,sscolumns])

print('Data scaled')
le=LabelEncoder()
le.fit(y)
Y_train=le.transform(y)

print('y labeled')

#fitting model
randomf=RandomForestClassifier()
randomf.fit(X,y)
print('model fitted')
#exporting model
joblib.dump(randomf, 'randomf_oneyear.pkl')
joblib.dump(imp_mean, 'imputer_oneyear.pkl')
joblib.dump(ss, 'standardscaler_oneyear.pkl')
print('model and scalers exported')

