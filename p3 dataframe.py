import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate Synthetic Data
np.random.seed(42)
n_samples = 1000

# Features
heartbeat = np.random.randint(60, 181, n_samples)  # Heartbeat (60-180 bpm)
temperature = np.random.uniform(36.0, 40.0, n_samples)  # Body temperature (36-40 °C)
hydration = np.random.uniform(0.5, 1.5, n_samples)  # Hydration level (arbitrary scale)
blood_pressure = np.random.randint(90, 180, n_samples)  # Blood pressure (90-180)

# Target: Injury (1 = yes, 0 = no)
injury = (heartbeat * 0.03 + temperature * 1.5 - hydration * 2 + blood_pressure * 0.02 + np.random.normal(0, 1, n_samples)) > 60
injury = injury.astype(int)

# Create a DataFrame
data = pd.DataFrame({
    "Heartbeat": heartbeat,
    "Temperature": temperature,
    "Hydration": hydration,
    "BloodPressure": blood_pressure,
    "Injury": injury
})

# Step 2: Split Data into Training and Testing Sets
X = data[["Heartbeat", "Temperature", "Hydration", "BloodPressure"]]
y = data["Injury"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Take Input from User and Make Predictions
print("\n--- Predict Injury Status ---")
heartbeat_input = float(input("Enter Heartbeat (bpm): "))
temperature_input = float(input("Enter Temperature (°C): "))
hydration_input = float(input("Enter Hydration Level (scale 0.5-1.5): "))
blood_pressure_input = float(input("Enter Blood Pressure (mmHg): "))

# Create a DataFrame for user input
user_data = pd.DataFrame({
    "Heartbeat": [heartbeat_input],
    "Temperature": [temperature_input],
    "Hydration": [hydration_input],
    "BloodPressure": [blood_pressure_input]
})

# Make prediction
prediction = model.predict(user_data)
prediction_label = "Injured" if prediction[0] == 1 else "Not Injured"

# Output the result
print(f"\nThe sports person is: {prediction_label}")
