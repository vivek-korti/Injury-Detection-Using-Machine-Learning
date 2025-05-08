import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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

# Step 3: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Make Predictions and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Generate Linear Graph
# Calculate probabilities for the test set
probabilities = model.predict_proba(X_test)[:, 1]

# Plot workload vs. probability of injury
plt.figure(figsize=(8, 6))
plt.scatter(X_test["Workload"], probabilities, color="blue", alpha=0.6, label="Predicted Probability")
plt.title("Workload vs Probability of Injury")
plt.xlabel("Workload")
plt.ylabel("Probability of Injury")
plt.legend()
plt.grid(True)
plt.show()
