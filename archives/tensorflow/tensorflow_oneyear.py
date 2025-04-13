import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Define columns
sscolumns = ['horse_prize_1y', 'horse_avg_km_time_6m',
             'horse_avg_km_time_12m', 'horse_min_km_time_6m',
             'horse_min_km_time_12m', 'horse_min_km_time_improve_12m',
             'horse_avg_km_time_improve_12m', 'horse_gals_1y',
             'horse_wins_1y', 'horse_podiums_1y', 'horse_fizetos_1y',
             'jockey_wins_1y', 'horse_wins_percent_1y',
             'horse_podiums_percent_1y', 'horse_fizetos_percent_1y']

categoricalcolumns = ['race_length', 'horse_age']

labelcolumns = ['horse_id', 'stable_id', 'jockey_id']

Xcolumns = sscolumns

# Load data
df = pd.read_csv(r"C:\Users\bence\projectderbiuj\data\merged_output.csv")

# Filter data
df = df[df["id"] > 146717]  # Filter for data after 2020
df = df[df['rank'] != 20]
df = df[df['time'] != 0]

# Handle missing values in competitor columns (if any)
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'].fillna(-1, inplace=True)  # Fill missing values with -1

# Apply Label Encoding to competitor columns
le = LabelEncoder()
for i in range(1, 14):  # For competitor_1 to competitor_14
    df[f'competitor_{i}'] = le.fit_transform(df[f'competitor_{i}'].astype(str))

# Apply Label Encoding to label columns
for col in labelcolumns:
    df[col] = le.fit_transform(df[col].astype(str))

# Drop rows where the target variable 'time' is NaN
df = df.dropna(subset=['time'])

y = df['rank']

# Assign X
X = df[Xcolumns + [f'competitor_{i}' for i in range(1, 14)] + labelcolumns]

# One-Hot Encode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
encoded = ohe.fit_transform(df[categoricalcolumns])
features = ohe.categories_
X = pd.concat([X, encoded], axis=1)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

# Impute missing values
imp_mean = SimpleImputer(strategy='mean')
X_train.loc[:, sscolumns] = imp_mean.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = imp_mean.transform(X_test.loc[:, sscolumns])

# Scale numerical features
scaler = StandardScaler()
X_train.loc[:, sscolumns] = scaler.fit_transform(X_train.loc[:, sscolumns])
X_test.loc[:, sscolumns] = scaler.transform(X_test.loc[:, sscolumns])

# Save preprocessing objects
joblib.dump(scaler, r"C:\Users\bence\projectderbiuj\models\scaler.pkl")
joblib.dump(features, r"C:\Users\bence\projectderbiuj\models\features_oneyear.pkl")
joblib.dump(imp_mean, r"C:\Users\bence\projectderbiuj\models\imputer.pkl")

# Define the TensorFlow model for classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Accuracy on Test Data: {accuracy}")

# Make predictions
Y_pred = (model.predict(X_test).flatten() > 0.5).astype(int)

# Calculate accuracy and classification report
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# Save the trained model
model.save(r"C:\Users\bence\projectderbiuj\models\best_classification_model_tensorflow.h5")
print("Model saved successfully.")

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()