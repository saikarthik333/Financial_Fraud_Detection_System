# 💳 Financial Fraud Detection System Using Clustering & Anomaly Detection

This project implements an unsupervised learning approach to detect anomalies in financial transactions. Using **K-Means** and **DBSCAN** clustering algorithms combined with **PCA** for dimensionality reduction, the system identifies unusual behavior that could represent fraudulent activity.

---

## 📌 Project Overview

- 🧠 **Techniques Used:** K-Means Clustering, DBSCAN, PCA (Principal Component Analysis), Elbow Method
- 📈 **Evaluation Metrics:** Silhouette Score (0.4718), Davies-Bouldin Index (0.8899)
- ⚠️ **Detected Anomalies:** 64 transactions marked as potential fraud
- 📊 **Tools & Libraries:** Python, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

---

## 📂 Dataset

- **Source:** Synthetic Financial Transactions Dataset
- **Sample Columns:**
  - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`
  - `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- **Preprocessing:** Removed string-based identifiers, label-encoded transaction types, standardized features.

---

## ⚙️ Methodology

### 🔹 1. Preprocessing
- Removed irrelevant features (e.g., names).
- Encoded categorical data (transaction type).
- Standardized numerical features for clustering.

### 🔹 2. K-Means Clustering
- **Elbow Method** used to determine the optimal number of clusters (k=4).
- Evaluated clustering quality using:
  - **Silhouette Score:** 0.4718
  - **Davies-Bouldin Index:** 0.8899

### 🔹 3. Anomaly Detection with DBSCAN
- Applied **DBSCAN** after reducing data dimensions with **PCA**.
- Identified noise points (`label = -1`) as anomalies.
- Detected **64 anomalies**, possibly fraudulent.

### 🔹 4. Visualization
- **Elbow Curve:** Helped choose optimal k for K-Means.
- **PCA Scatter Plot:** Visualized clusters and outliers from DBSCAN.

---

## 📊 Results & Analysis

| Method        | Metric               | Value     |
|---------------|----------------------|-----------|
| K-Means       | Silhouette Score     | 0.4718    |
| K-Means       | Davies-Bouldin Index | 0.8899    |
| DBSCAN        | Anomalies Detected   | 64        |

- **K-Means** provided moderately good clustering.
- **DBSCAN** effectively found density-based outliers, key to fraud detection.
- PCA helped in visualizing the clusters clearly in 2D.

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

---

## 🧪 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/saikarthik333/Financial_Fraud_Detection_System.git
   cd Financial_Fraud_Detection_System
