import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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

# Step 3: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Generate Linear Graph
# Plot injured and not injured data points
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="Workload", y="Fatigue", hue="Injury", palette="coolwarm", alpha=0.7)
plt.title("Workload and Fatigue: Injury vs No Injury")
plt.xlabel("Workload")
plt.ylabel("Fatigue")
plt.legend(title="Injury", labels=["Not Injured", "Injured"])

# Step 7: Add Linear Regression Line
# Fit linear regression for visual representation
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Calculate predictions for plotting
x_range = np.linspace(data["Workload"].min(), data["Workload"].max(), 100)
y_range = lin_reg.predict(pd.DataFrame({"Workload": x_range, "Fatigue": [data["Fatigue"].mean()] * len(x_range)}))

# Plot the regression line
plt.plot(x_range, y_range * 100, color="black", linestyle="--", label="Regression Line")
plt.legend()
plt.show()
