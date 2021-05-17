### Aaron Hiller, Kit Sloan
from sklearn.cluster import KMeans
import numpy as np
import cv2

n_clusters = 3

photo = cv2.imread('house.jpg')
w, h, _ = photo.shape
img = photo.reshape(w * h, 3)
kmeans = KMeans(n_clusters=n_clusters).fit(img)

# find out which cluster each pixel belongs to
labels = kmeans.predict(img)

# finds centroid of cluster
centroid = np.array(kmeans.cluster_centers_)

# recolor the image based on clustering
clustered_img = np.copy(img)
for i in range(len(clustered_img)):
    if labels[i] == 0:
        clustered_img[i] = [255, 0, 0]
    elif labels[i] == 1:
        clustered_img[i] = [0, 255, 0]
    else:
        clustered_img[i] = [0, 0, 255]

# reshape for display
clustered_img = clustered_img.reshape(w, h, 3)
cv2.imshow('Original', photo)
cv2.imshow('Clustered RGB', clustered_img)
print("Press any key to close the window")
key = cv2.waitKey()




