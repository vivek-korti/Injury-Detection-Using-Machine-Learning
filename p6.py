import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Pre-defined Dataset of Normal State Data
np.random.seed(42)
n_samples = 1000

# Generate normal state data
normal_heartbeat = np.random.randint(60, 100, n_samples)  # Normal heartbeat range
normal_spin = np.random.uniform(0.0, 10.0, n_samples)  # Spin in rad/s
normal_speed = np.random.uniform(0.0, 7.0, n_samples)  # Speed in m/s
normal_direction = np.random.uniform(0.0, 360.0, n_samples)  # Direction in degrees

# Target: Safe (0), Harmed (1), Injured (2)
labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])  # Simulated labels

# Create a DataFrame
data = pd.DataFrame({
    "Heartbeat": normal_heartbeat,
    "SpinMoment": normal_spin,
    "Speed": normal_speed,
    "Direction": normal_direction,
    "Label": labels  # 0 = Safe, 1 = Harmed, 2 = Injured
})

# Step 2: Train-Test Split
X = data[["Heartbeat", "SpinMoment", "Speed", "Direction"]]
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, multi_class="ovr")
model.fit(X_train, y_train)

# Step 4: Evaluate Model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Take Input Data for Collision
print("\n--- Enter Data for Two Players ---")

# Player 1 data
print("Player 1:")
heartbeat1 = float(input("Enter Heartbeat: "))
spin1 = float(input("Enter Spin Moment (rad/s): "))
speed1 = float(input("Enter Speed (m/s): "))
direction1 = float(input("Enter Direction (degrees): "))

# Player 2 data
print("\nPlayer 2:")
heartbeat2 = float(input("Enter Heartbeat: "))
spin2 = float(input("Enter Spin Moment (rad/s): "))
speed2 = float(input("Enter Speed (m/s): "))
direction2 = float(input("Enter Direction (degrees): "))

# Step 6: Predict Injury Status for Each Player
player_data = pd.DataFrame({
    "Heartbeat": [heartbeat1, heartbeat2],
    "SpinMoment": [spin1, spin2],
    "Speed": [speed1, speed2],
    "Direction": [direction1, direction2]
})

predictions = model.predict(player_data)
prediction_labels = ["Safe", "Harmed", "Injured"]

# Output Results
print("\n--- Results ---")
print(f"Player 1: {prediction_labels[predictions[0]]}")
print(f"Player 2: {prediction_labels[predictions[1]]}")
