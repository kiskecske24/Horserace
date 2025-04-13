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


sscolumns=['horse_avg_km_time_6m',
       'horse_min_km_time_6m',
        'horse_gals_1y', 'horse_wins_1y',
       'horse_podiums_1y','horse_fizetos_1y','jockey_wins_1y',
       'horse_wins_percent_1y','horse_podiums_percent_1y', 'horse_fizetos_percent_1y',
       'jockey_wins_1y','horse_wins_percent_1y',
       'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns=['number','race_length','jockey_id','stable_id','trainer_id']

Xcolumns=['horse_avg_km_time_6m',
       'horse_min_km_time_6m',
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
    df.to_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)



df=pd.read_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv")

X=df[Xcolumns]

dummies=True

#Assigning X

def getdummies(df, X):
       
       ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
       encoded = ohe.fit_transform(df[categoricalcolumns])
       features = ohe.categories_
       print('Got dummies')
       X=pd.concat([X,encoded], axis=1)
       print('Assigned X')
       return X, features

#Assigning y

y=df['rank']
print('Assigned y')

#splitting data

if dummies==True:
       X, features = getdummies(df, X)
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
else:
       X_train, X_test, Y_train, Y_test=train_test_split(X,y, test_size=0.2, shuffle=False)
       features=None

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

best_model = MLPClassifier(random_state=1)
best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)
print('model fitted')

#exporting model

joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\modelmlp_6months.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_6months.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_6months.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_6months.pkl")
print('model and scalers exported')

#plotting data

print(Y_test)
print(Y_pred)
print('R2score: ' , r2_score(Y_test, Y_pred))
print('Accurace score: ', accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Feature importance

importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\mlpfeature_importance_6months.csv", index=False)