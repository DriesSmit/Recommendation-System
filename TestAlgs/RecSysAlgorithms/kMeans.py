#https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

#http://www.data-mania.com/blog/recommendation-system-python/
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# First let's create a dataset called X, with 6 records and 2 features each.
X = np.array([[-1, -1], [-2, -4], [-2, -1], [-1, 4], [-2, 5], [-1, 6], [4, 0], [3, 1], [4, -1]])

# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
kmeans = KMeans(n_clusters=3).fit(X)
centres = kmeans.cluster_centers_
# Plot also the training points

point = [[-4,-5]]
plt.plot(point[0][0],point[0][1],'bo')

print("Centres: ",centres)
print(kmeans.predict(point))

plt.scatter(X[:,0],X[:,1],c='red')
plt.plot(centres[:,0],centres[:,1],'go')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.show()

