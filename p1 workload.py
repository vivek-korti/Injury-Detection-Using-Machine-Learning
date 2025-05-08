
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Synthetic Data
np.random.seed(42)
n_samples = 1000

# Features
workload = np.random.randint(1, 101, n_samples)  # Workload level (1-100)
fatigue = np.random.randint(1, 101, n_samples)  # Fatigue level (1-100)

# Target: Injury (1 = yes, 0 = no)
injury = (workload * 0.5 + fatigue * 0.7 + np.random.normal(0, 10, n_samples)) > 100
injury = injury.astype(int)

# Create a DataFrame
data = pd.DataFrame({
    "Workload": workload,
    "Fatigue": fatigue,
    "Injury": injury
})

# Step 2: Split Data into Training and Testing Sets
X = data[["Workload", "Fatigue"]]
y = data["Injury"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Injury", "Injury"], yticklabels=["No Injury", "Injury"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Step 6: Test the Model with New Data
new_data = pd.DataFrame({
    "Workload": [80, 20, 50],
    "Fatigue": [90, 10, 60]
})

predictions = model.predict(new_data)
print("Predictions for new data:")
print(new_data.assign(Injury=predictions))
