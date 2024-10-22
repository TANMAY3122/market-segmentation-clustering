# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Step 2: Data Preprocessing
# Dropping unnecessary columns (CustomerID)
df = data.drop(['CustomerID'], axis=1)

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Step 3: PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Creating a DataFrame with PCA components
pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])

# Step 4: Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(pca_df)

# Adding the cluster labels to the DataFrame
pca_df['Cluster'] = kmeans_clusters

# Visualizing K-Means Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set1', s=100)
plt.title('Customer Segmentation Using K-Means Clustering')
plt.show()

# Step 5: Extension - Applying DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(pca_df[['PCA1', 'PCA2']])

# Adding DBSCAN cluster labels to the DataFrame
pca_df['DBSCAN_Cluster'] = dbscan_clusters

# Visualizing DBSCAN Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='DBSCAN_Cluster', data=pca_df, palette='plasma', s=100)
plt.title('Customer Segmentation Using DBSCAN Clustering')
plt.show()

# Step 6: Extension - Applying Hierarchical Clustering
linked = linkage(pca_df[['PCA1', 'PCA2']], method='ward')

# Plot the dendrogram
plt.figure(figsize=(10,7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

# Applying Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
agglo_clusters = agglo.fit_predict(pca_df[['PCA1', 'PCA2']])

# Adding Agglomerative Clustering labels to the DataFrame
pca_df['Agglo_Cluster'] = agglo_clusters

# Visualizing Hierarchical Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Agglo_Cluster', data=pca_df, palette='rainbow', s=100)
plt.title('Customer Segmentation Using Hierarchical Clustering')
plt.show()

# Display final dataset with cluster labels
print(pca_df.head())
