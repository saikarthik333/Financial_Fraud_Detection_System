import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

file_path = '/content/Synthetic_Financial_datasets_log.csv'
data = pd.read_csv(file_path)

data = data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
data['type'] = data['type'].astype('category').cat.codes

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=['isFraud']))

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(scaled_data)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

dbscan = DBSCAN(eps=0.5, min_samples=5)
data['dbscan_cluster'] = dbscan.fit_predict(pca_data)

kmeans_silhouette = silhouette_score(scaled_data, data['kmeans_cluster'])
kmeans_davies_bouldin = davies_bouldin_score(scaled_data, data['kmeans_cluster'])

print("K-means Silhouette Score:", kmeans_silhouette)
print("K-means Davies-Bouldin Index:", kmeans_davies_bouldin)

data['anomaly'] = data['dbscan_cluster'].apply(lambda x: 1 if x == -1 else 0)


print("Number of anomalies detected:", data['anomaly'].sum())

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['dbscan_cluster'], cmap='viridis', marker='o', s=10)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('DBSCAN Clustering and Anomalies')
plt.show()

data.to_csv('clustering_anomaly_detection_results.csv', index=False)
print("Results saved to clustering_anomaly_detection_results.csv")
