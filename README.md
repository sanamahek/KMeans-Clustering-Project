# 📊 K-Means Clustering with Education-Level Visualization

This project demonstrates **K-Means clustering** on synthetic employee data, grouping employees by **Years of Experience**, **Monthly Salary**, and **Education Level**.  
It includes centroid visualization, silhouette score evaluation, and a cluster summary table.

---

## 🚀 Features
- Generates synthetic employee dataset (Age, Education, Experience, Salary, Performance).
- Encodes categorical variables (Education Level).
- Applies **K-Means clustering** with 3 clusters.
- Calculates **Silhouette Score** for cluster quality.
- Visualizes clusters:
  - Color → Cluster
  - Marker → Education Level
  - Centroid → Black "X"
- Provides a **Cluster vs Education Distribution** summary.

---

## 🛠️ Requirements
Install the following Python libraries before running:

```bash
pip install numpy pandas matplotlib scikit-learn
