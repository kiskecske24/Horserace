import sqlite3
import pandas as pd
import numpy as np

def getdata():
    try:
        # Connect to the database
        conn = sqlite3.connect(r"C:\Users\bence\projectderbiuj\data\trotting1012.db")
        
        # SQL query
        query = """
        SELECT 
            r.id AS race_id,
            h.id AS horse_id
        FROM races r
        CROSS JOIN horses h
        LEFT JOIN horse_races hr ON hr.race_id = r.id AND hr.horse_id = h.id
        WHERE hr.horse_id IS NOT NULL
        """
        
        # Execute the query and load the result into a DataFrame
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Check if the DataFrame is empty
        if df.empty:
            print("The query returned no results.")
            return
        
        # Group by race_id and collect horse IDs into lists
        grouped = df.groupby('race_id')['horse_id'].apply(list).reset_index()
        
        # Expand the grouped DataFrame to include competitor columns
        expanded_rows = []
        for _, row in grouped.iterrows():
            race_id = row['race_id']
            horse_ids = row['horse_id']
            for horse_id in horse_ids:
                # Exclude the current horse_id from the competitors
                competitors = [comp for comp in horse_ids if comp != horse_id]
                # Limit to 14 competitors
                competitors = competitors[:14]
                # Create a row with race_id, horse_id, and competitor columns
                expanded_rows.append({
                    'race_id': race_id,
                    'horse_id': horse_id,
                    **{f'competitor_{i+1}': competitors[i] if i < len(competitors) else None for i in range(14)}
                })
        
        # Create a new DataFrame from the expanded rows
        expanded_df = pd.DataFrame(expanded_rows)
        
        # Load the original CSV
        original_csv_path = r"C:\Users\bence\projectderbiuj\data\querynewtop4.csv"
        original_df = pd.read_csv(original_csv_path)
        
        # Join the expanded DataFrame with the original CSV on `race_id` and `horse_id`
        merged_df = pd.merge(original_df, expanded_df, on=['race_id', 'horse_id'], how='left')
        
        # Save the merged DataFrame to a new CSV file
        output_path = r"C:\Users\bence\projectderbiuj\data\merged_output.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

getdata()
print('Data fetched and merged successfully')