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
from sklearn.neural_network import MLPClassifier


sscolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns=['number','race_length','horse_id','stable_id','jockey_id']

Xcolumns=['horse_prize_1y', 'horse_avg_km_time_6m',
       'horse_avg_km_time_12m', 'horse_min_km_time_6m',
       'horse_min_km_time_12m',
       'horse_min_km_time_improve_12m',
       'horse_avg_km_time_improve_12m',
       'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

#getting data
def getdata():
    conn = sqlite3.connect("trottingnew1012.db")
    query = "SELECT * FROM horse_races_aggregated WHERE race_id>146717"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(r"C:\Users\bence\OneDrive\projectderbiuj\querynew2020.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)



df=pd.read_csv('querynew1012withtop4.csv')

#getting dummies


#df.to_csv(r"C:\Users\bence\OneDrive\Project derbi\lasttestwithdummies.csv", index=False)

#reading data

#df=df.drop(df[df['rank']==20].index)
df = df[df["id"]>146717]
df = df[df["id"]<161944]
print(df.head())

print('Reading csv')
#Assigning X
X=df[Xcolumns]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
features = ohe.categories_
print('Got dummies')
X=pd.concat([X,encoded], axis=1)
print(X.head())
print('Assigned X')
#Assigning y
y=df['top4']
print('Assigned y')
#splitting data
X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
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
model_fit = MLPClassifier(random_state=1, max_iter=10)
model_fit.fit(X_train, Y_train)
Y_pred = model_fit.predict(X_test)
print('model fitted')
#exporting model
joblib.dump(model_fit, 'modelrandomf_oneyear.pkl')
joblib.dump(imp_mean, 'imputer_oneyear.pkl')
joblib.dump(ss, 'standardscaler_oneyear.pkl')
joblib.dump(features,'features_oneyear.pkl')
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


