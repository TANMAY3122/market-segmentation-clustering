# Market Segmentation Using Clustering

This project focuses on applying various **clustering techniques** for market segmentation using **unsupervised learning**. We use the **Mall Customer Segmentation Data** from Kaggle, and techniques like **K-Means Clustering**, **DBSCAN**, and **Hierarchical Clustering** to group customers based on purchasing behavior and demographics.

## Overview
**Market Segmentation** is the process of dividing a broad consumer or business market into sub-groups of consumers based on shared characteristics such as demographics, spending behavior, etc. In this project, we use various **clustering algorithms** to segment customers, helping businesses tailor strategies to different groups.

### Goal
The goal of this project is to:
- Group customers into clusters based on similarities in spending behavior and demographics.
- Apply **PCA** to reduce dimensionality and visualize the clusters.
- Interpret results to suggest actionable business strategies for each segment.

## Dataset
The **Mall Customer Segmentation Data** is used in this project. It contains 200 entries and the following columns:
- **CustomerID**: Unique ID for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousands.
- **Spending Score (1-100)**: Score assigned by the mall based on customer spending patterns.

You can find the full notebook and detailed analysis on Kaggle by following this link:
[Mall Customer Segmentation & Clustering Analysis](https://www.kaggle.com/code/gadigevishalsai/mall-customer-segmentation-clustering-analysis#Mall-Customer-Segmentation-Data---Clustering-and-Analysis)

## Clustering Techniques
We use the following clustering algorithms to perform market segmentation:

### 1. K-Means Clustering
**K-Means** is a popular algorithm that partitions the dataset into K clusters by minimizing intra-cluster variance. In this project:
- We use **K-Means** with `n_clusters=5` and visualize the clusters in a 2D plot after applying PCA.
  
### 2. DBSCAN Clustering
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that identifies clusters based on density, rather than predefined centroids.
- We apply DBSCAN with `eps=0.5` and `min_samples=5` to discover natural clusters and outliers.

### 3. Hierarchical Clustering
**Hierarchical Clustering** builds a hierarchy of clusters and uses dendrograms to visualize them. It is useful for understanding the structure of the data.
- We apply **Agglomerative Clustering** with `n_clusters=5` and use the **ward** linkage criterion.

## Principal Component Analysis (PCA)
Before applying clustering, we use **PCA** to reduce the dimensionality of the dataset from 3D to 2D. This allows us to visualize the clusters and makes the algorithms more efficient.

## Results
After applying the clustering techniques, we obtained the following insights:

- **K-Means** identified 5 distinct clusters based on customer spending behavior and demographics.
- **DBSCAN** revealed outliers in the dataset, potentially identifying niche or VIP customers.
- **Hierarchical Clustering** provided a clear hierarchy of clusters, helping us understand the relationships between different customer segments.

### Visualizations
- **K-Means Clustering**:  
  ![KMeans](images/kmeans_clusters.png)

- **DBSCAN Clustering**:  
  ![DBSCAN](images/dbscan_clusters.png)

- **Hierarchical Clustering Dendrogram**:  
  ![Dendrogram](images/hierarchical_dendrogram.png)

## Conclusion
The project demonstrates how **clustering** techniques can be used for **market segmentation**. The key takeaways include:
- Different clustering algorithms have their strengths. **K-Means** works well with well-separated clusters, **DBSCAN** detects outliers, and **Hierarchical Clustering** provides a deeper understanding of cluster relationships.
- Reducing dimensions with **PCA** makes the clustering process more efficient and allows for easier visualization of the results.

## Future Enhancements
Potential improvements for this project include:
- Experimenting with different feature selection techniques to enhance clustering performance.
- Adding more customer behavior metrics (e.g., purchase frequency) to create richer customer profiles.
- Implementing cluster-specific business strategies for real-world application.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
