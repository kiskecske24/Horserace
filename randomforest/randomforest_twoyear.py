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


from pathlib import Path

project_root = model_dir = Path(__file__).resolve().parent.parent


sscolumns=['horse_prize_1y', 'horse_prize_2y',
       'horse_avg_km_time_2y', 'horse_avg_km_time_3y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m', 'horse_min_km_time_2y',
       'horse_min_km_time_improve_12m', 'horse_min_km_time_improve_2y',
       'horse_avg_km_time_improve_12m', 'horse_avg_km_time_improve_2y',
       'horse_gals_1y', 'horse_gals_2y','horse_wins_1y',
       'horse_wins_2y','horse_podiums_1y',
       'horse_podiums_2y','horse_fizetos_1y',
       'horse_fizetos_2y','jockey_wins_1y',
       'jockey_wins_2y', 'horse_wins_percent_1y',
       'horse_wins_percent_2y',
       'horse_podiums_percent_1y', 'horse_podiums_percent_2y','horse_fizetos_percent_1y',
       'horse_fizetos_percent_2y']

categoricalcolumns=['number','race_length','horse_id','jockey_id','stable_id']

Xcolumns=['horse_prize_1y', 'horse_prize_2y',
       'horse_avg_km_time_2y', 'horse_avg_km_time_3y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m', 'horse_min_km_time_2y',
       'horse_min_km_time_improve_12m', 'horse_min_km_time_improve_2y',
       'horse_avg_km_time_improve_12m', 'horse_avg_km_time_improve_2y',
       'horse_gals_1y', 'horse_gals_2y','horse_wins_1y',
       'horse_wins_2y','horse_podiums_1y',
       'horse_podiums_2y','horse_fizetos_1y',
       'horse_fizetos_2y','jockey_wins_1y',
       'jockey_wins_2y', 'horse_wins_percent_1y',
       'horse_wins_percent_2y',
       'horse_podiums_percent_1y', 'horse_podiums_percent_2y','horse_fizetos_percent_1y',
       'horse_fizetos_percent_2y']





#getting data
def getdata():
    conn = sqlite3.connect(project_root / "data" / "trottingnew1012.db")
    query = "SELECT * FROM horse_races_aggregated WHERE race_id>146717"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(project_root / "data" / "querynew2020.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)



df=pd.read_csv('querynewtop5.csv')

#getting dummies


#df.to_csv(r"C:\Users\bence\OneDrive\Project derbi\lasttestwithdummies.csv", index=False)

#reading data
#df=pd.read_csv('lasttestwithdummies.csv')
df=df.drop(df[df['rank']==20].index)
df = df[df["id"]>146717]

#earlier ids:146717,10496

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
#splitting data
X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.001, shuffle=False)
print('Data splitted')
#preprocessing data
imp_mean = SimpleImputer(strategy='mean')
X_train.loc[:,sscolumns] = imp_mean.fit_transform(X_train.loc[:,sscolumns])
X_test.loc[:,sscolumns] = imp_mean.transform(X_test.loc[:,sscolumns])
print('Data imputed')
ss=preprocessing.StandardScaler()
X_train.loc[:,sscolumns]=ss.fit_transform(X_train.loc[:,sscolumns])
X_test.loc[:,sscolumns]=ss.transform(X_test.loc[:,sscolumns])
print('Data scaled')
le=LabelEncoder()
le.fit(Y_train)
Y_train=le.transform(Y_train)
Y_test=le.transform(Y_test)
print('y labeled')

#fitting model
model_fit = RandomForestClassifier(random_state=1, n_estimators=10, max_depth=10000)
model_fit.fit(X_train, Y_train)
Y_pred = model_fit.predict(X_test)
print('model fitted')
#exporting model
models_root = project_root / "models"

joblib.dump(model_fit, models_root / 'modelmlp_oneyear.pkl')
joblib.dump(imp_mean, models_root / 'imputer_oneyear.pkl')
joblib.dump(ss, models_root / 'standardscaler_oneyear.pkl')
joblib.dump(features, models_root / 'features_oneyear.pkl')
print('model and scalers exported')
#plotting data
print(Y_test)
print(Y_pred)
print('R2score: ' , r2_score(Y_test, Y_pred))
print('Accuracy score: ', accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()