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
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report



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
    df.to_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)



df=pd.read_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv")


#reading data

df=df.drop(df[df['rank']==20].index)
df = df[df["id"]>146717]
df = df[df["id"]<161945]

print('Reading csv')

#Assigning X
X=df[Xcolumns]
dummies=True

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

best_model=linear_model.LogisticRegression()


# Fit the best model
best_model.fit(X_train, Y_train)
Y_pred = best_model.predict(X_test)

# Evaluate the model

print('R2score: ', r2_score(Y_test, Y_pred))


# Plotting Actual vs Predicted
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Exporting the model and preprocessing objects
joblib.dump(best_model, r"C:\Users\bence\projectderbiuj\models\logistic_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer_oneyear.pkl")
joblib.dump(ss, r"C:\Users\bence\projectderbiuj\models\standardscaler_oneyear.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
print('Model and scalers exported')

# Feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
feature_importance_df.to_csv(r"C:\Users\bence\projectderbiuj\data\logisticfeature_importance_oneyear.csv", index=False)