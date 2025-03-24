<h1>Financial Fraud Detection System Using Clustering and Anomaly Detection<h1>
  
<h2>Overview<h2>

This project implements a financial fraud detection system using unsupervised machine learning techniques. By leveraging clustering algorithms like K-means and DBSCAN, the system identifies anomalous transactions that may indicate fraudulent activity. Dimensionality reduction using Principal Component Analysis (PCA) enhances clustering performance and visualization, while data processing is efficiently handled with NumPy and Pandas. The project is evaluated using metrics such as the Silhouette Score (0.4718) and Davies-Bouldin Index (0.8899), and visual insights are generated with Matplotlib and Seaborn.

<h3>Features<h3>
Financial Fraud Detection: Identify fraudulent transactions using unsupervised clustering.

K-means Clustering: Partition transaction data into optimal clusters (k = 4 determined via the Elbow Method).

DBSCAN Anomaly Detection: Detect outliers based on density, flagging anomalies in the dataset.

Dimensionality Reduction: Utilize PCA to reduce data complexity for improved clustering and visualization.

Performance Evaluation: Measure clustering quality with the Silhouette Score and Davies-Bouldin Index.

Data Visualization: Generate insightful plots to illustrate clusters and highlight anomalies.
