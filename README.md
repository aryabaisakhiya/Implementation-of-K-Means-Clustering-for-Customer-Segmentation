# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary python libraries.
2.Pick customer segment quantity (k).
3.Seed cluster centers with random data points.
4.Assign customers to closest centers.
5.Re-center clusters and repeat until stable.
```

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Arya Baisakhiya
RegisterNumber:212222040019
*/
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers.csv")
data
x=data[['Annual Income (k$)','Spending Score (1-100)']]
x
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(x)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','b','g','c','m']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
              color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![322861315-ef9dc3ac-259a-4324-b6c2-1d39574fdc88](https://github.com/aryabaisakhiya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393645/558bf884-9352-4da3-858b-5cf4a6cb6253)
![322861542-c1cb9b62-3021-4b7a-b36a-d2a949c466e4](https://github.com/aryabaisakhiya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393645/87705726-a786-4f27-8a68-0a69b267a68e)
![322861693-9ebb181d-3d26-4686-adf2-00a486b8f868](https://github.com/aryabaisakhiya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119393645/b861bbab-67ea-4258-8f8d-799fffa65687)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
