# K-MEANS CLUSTERING WITH
# CLUSTER NUMBER + EDUCATION LEVEL + CENTROID VISUALIZATION
# ========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ----------------------------------------------
# 1️⃣ CREATE COMPANY DATA
# ----------------------------------------------
np.random.seed(42)
n = 60

df = pd.DataFrame({
    "Employee_ID": range(1, n + 1),
    "Age": np.random.randint(20, 50, n),
    "Education_Level": np.random.choice(
        ["High School", "Diploma", "Graduate"],
        n,
        p=[0.3, 0.35, 0.35]
    ),
    "Years_of_Experience": np.random.randint(0, 25, n),
    "Monthly_Salary": np.random.randint(15000, 90000, n),
    "Performance_Score": np.random.randint(40, 100, n)
})

# ----------------------------------------------
# 2️⃣ PREPROCESSING
# ----------------------------------------------
df_encoded = pd.get_dummies(df, columns=["Education_Level"], drop_first=True)

X = df_encoded.drop("Employee_ID", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------
# 3️⃣ APPLY K-MEANS
# ----------------------------------------------
k = 3
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Silhouette Score:",
      silhouette_score(X_scaled, df["Cluster"]))

# ----------------------------------------------
# 🔹 GET CENTROIDS (BACK TO ORIGINAL SCALE)
# ----------------------------------------------
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(centroids, columns=X.columns)
centroids_df["Cluster"] = range(k)

# ----------------------------------------------
# 4️⃣ CLUSTER VISUALIZATION
# Color → Cluster
# Marker → Education Level
# Centroid → Black X
# ----------------------------------------------
plt.figure(figsize=(9, 6))

cluster_colors = {0: "red", 1: "blue", 2: "green"}
edu_markers = {
    "High School": "o",
    "Diploma": "s",
    "Graduate": "^"
}

for cluster in range(k):
    for edu in edu_markers:
        subset = df[
            (df["Cluster"] == cluster) &
            (df["Education_Level"] == edu)
        ]
        plt.scatter(
            subset["Years_of_Experience"],
            subset["Monthly_Salary"],
            color=cluster_colors[cluster],
            marker=edu_markers[edu],
            label=f"Cluster {cluster} - {edu}"
        )

# 🔹 Plot Centroids
plt.scatter(
    centroids_df["Years_of_Experience"],
    centroids_df["Monthly_Salary"],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.xlabel("Years of Experience")
plt.ylabel("Monthly Salary")
plt.title("K-Means Clusters with Education Levels & Centroids")
plt.legend(fontsize=8)
plt.grid(True)
plt.show()

# ----------------------------------------------
# 5️⃣ CLUSTER SUMMARY
# ----------------------------------------------
summary = df.groupby(["Cluster", "Education_Level"]).size().unstack(fill_value=0)
print("\nCluster vs Education Distribution:")
print(summary)
