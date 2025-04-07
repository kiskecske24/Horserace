import sqlite3
import pandas as pd
import numpy as np
def getdata():
    conn = sqlite3.connect(r"C:\Users\bence\projectderbiuj\data\trotting1012.db")
    query = "SELECT * FROM horse_races_aggregated"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv", index=False)
    df.drop(df.loc[df['rank']==0].index, inplace=True)
getdata()
print("Data fetched and saved to CSV.")