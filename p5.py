import numpy as np
import pandas as pd
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

# Step 2: Generate Plots

# 1. Scatter Plot: Workload vs Fatigue, colored by Injury
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="Workload", y="Fatigue", hue="Injury", palette="coolwarm", alpha=0.7)
plt.title("Scatter Plot: Workload vs Fatigue")
plt.xlabel("Workload")
plt.ylabel("Fatigue")
plt.legend(title="Injury", labels=["Not Injured", "Injured"])
plt.show()

# 2. Histogram: Distribution of Workload and Fatigue
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=data, x="Workload", hue="Injury", kde=True, palette="coolwarm", alpha=0.6)
plt.title("Histogram: Workload Distribution")
plt.xlabel("Workload")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(data=data, x="Fatigue", hue="Injury", kde=True, palette="coolwarm", alpha=0.6)
plt.title("Histogram: Fatigue Distribution")
plt.xlabel("Fatigue")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Boxplot: Workload and Fatigue by Injury
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=data, x="Injury", y="Workload", palette="coolwarm")
plt.title("Boxplot: Workload by Injury")
plt.xlabel("Injury (0 = Not Injured, 1 = Injured)")
plt.ylabel("Workload")

plt.subplot(1, 2, 2)
sns.boxplot(data=data, x="Injury", y="Fatigue", palette="coolwarm")
plt.title("Boxplot: Fatigue by Injury")
plt.xlabel("Injury (0 = Not Injured, 1 = Injured)")
plt.ylabel("Fatigue")
plt.tight_layout()
plt.show()

# 4. Pairplot: Workload, Fatigue, and Injury Relationships
sns.pairplot(data, vars=["Workload", "Fatigue"], hue="Injury", palette="coolwarm", diag_kind="kde")
plt.suptitle("Pairplot: Feature Relationships", y=1.02)
plt.show()
