# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:04:37 2022

@author: roigu
"""
from keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
import scipy
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin , pairwise_distances_argmin_min
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_train.reshape(len(x_train),-1)
Y = y_train

X_train = X.astype(np.float32) / 255.

#Fitting every label from the model to the real one 
def retrieve_info(cluster_labels,y_train):

# Initializing
    reference_labels = {}
# For loop to run through each label of cluster label
    for i in range(len(np.unique(cluster_labels))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels


#Function created to define centroids according to an article

def initial_centroids_x(data,n):
    centroids = np.zeros((n,data.shape[1]))
    i = LA.norm(data,axis = 1)
    i = np.argmax(i)
    centroids[0] = data[i]
    data = data[np.r_[0:i , i+1:data.shape[0]]]
    tmp_c = centroids[0].reshape(1,-1)
    dist = pairwise_distances(tmp_c,data)
    inx = np.argmax(dist)
    centroids[1] = data[inx]
    data = data[np.r_[0:inx , inx+1:data.shape[0]]]
    for i in range(2,n):
        dist = pairwise_distances(centroids[0:i],data)
        arr_inx = np.zeros((dist.shape[0],1))
        arr_dist  = np.min(dist,axis = 0)
        arr_inx  = np.argmin(dist,axis = 0)
        arr_inx = arr_inx.reshape(-1,1)
        inx_min = int(arr_inx[np.argmax(arr_dist)][0])
        centroids[i] = data[inx_min]
        data = data[np.r_[0:inx_min , inx_min+1:data.shape[0]]]
    return centroids
# Ordinary Kmeans
def K_means(X, n_clusters,pretrained_clusters = None,rseed=2):
    # 1. Randomly choose clusters
    if pretrained_clusters is not None:
        centers = pretrained_clusters
    else:
        rng = np.random.RandomState(rseed)
        i = rng.permutation(X.shape[0])[:n_clusters]
        centers = X[i]
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels


def initial_centroids_max_var(data,n_clusters):
    n_features = data.shape[1]
    var_array = np.zeros((n_features,1))
    for i in range (n_features):
        var_array[i] = np.std(data[:,i])**2
    var_max = np.argmax(var_array)
    data_var_max = data[:,var_max]
    n_examples = data.shape[0]
    data_var_max = np.argsort(data_var_max)
    num_per_group = np.floor(n_examples/n_clusters)
    groups = np.zeros((n_clusters,int(num_per_group),n_features))
    for i in range (n_clusters):
        groups[i,:,:] = data[data_var_max[int(i*num_per_group):int((i+1)*num_per_group)]]
    centroids = np.mean(groups,axis = 1)
    return centroids
# My improved Kmeans
def improved_kmeans(X, n_clusters,pretrained_clusters = None, rseed=2):
    # 1. Randomly choose clusters
    if (pretrained_clusters is not None):
        centers = pretrained_clusters
    else:        
        rng = np.random.RandomState(rseed)
        i = rng.permutation(X.shape[0])[:n_clusters]
        centers = X[i]
    cluster , dif = pairwise_distances_argmin_min(X, centers)

    while True:
        # 2a. Assign labels based on closest center
        dist_orig = pairwise_distances(X, centers)

        for i in range (X.shape[0]):
            #if pairwise_distances(X[i].reshape(1,-1),centers[cluster[i]].reshape(1,-1)) > dif[i]:
            if dist_orig[i,cluster[i]] > dif[i]:
                cluster[i] , dif[i] =  pairwise_distances_argmin_min(X[i].reshape(1,-1),centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[cluster == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, cluster
#Centroid initialization functions
num_centers = [10,20,50,100,150]
for i in (num_centers):
    start = time.time()
    centers, labels = K_means(X_train, i)
    end = time.time()
    print('my original Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers, labels = improved_kmeans(X_train, i)
    end = time.time()
    print('improved Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers_init = initial_centroids_max_var(X_train,i)
    centers, labels = K_means(X_train, i,centers_init)
    end = time.time()
    print('my original Kmeans algorithm with max_var function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with max_var function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers_init = initial_centroids_max_var(X_train,i)
    centers, labels = improved_kmeans(X_train, i,centers_init)
    end = time.time()
    print('improved Kmeans algorithm with max_var function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with max_var function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers_init = initial_centroids_x(X_train,i)
    centers, labels = K_means(X_train, i,centers_init)
    end = time.time()
    print('my original Kmeans algorithm with max_var function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with max_var function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers_init = initial_centroids_x(X_train,i)
    centers, labels = improved_kmeans(X_train, i,centers_init)
    end = time.time()
    print('improved Kmeans algorithm with max_var function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with max_var function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_train)))
PCA_components = [10,20,50,100,200]
num_clusters = 20
start = time.time()
centers, labels = K_means(X_train, num_clusters)
end = time.time()
print('my original Kmeans algorithm with random centroid initialization running time for original data is ' 
      + str( - start + end) + ' sec')
reference_labels = retrieve_info(labels,y_train)
number_labels = np.random.rand(len(labels))
for j in range(len(labels)):
        number_labels[j] = reference_labels[labels[j]]
print('my original Kmeans algorithm with random centroid initialization running time for original data is ' 
      + str(accuracy_score(number_labels,y_train)))
start = time.time()
centers, labels = improved_kmeans(X_train, num_clusters)
end = time.time()
print('improved Kmeans algorithm with random centroid initialization running time is ' + str( - start + end) + ' sec')
reference_labels = retrieve_info(labels,y_train)
number_labels = np.random.rand(len(labels))
for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
print('improved Kmeans algorithm with random centroid initialization accuracy is ' + str(accuracy_score(number_labels,y_train)))
#PCA for Mnist
for i in (PCA_components):
    start = time.time()
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X_train) 
    centers, labels = K_means(X_pca, num_clusters)
    end = time.time()
    print('my original Kmeans algorithm running time for ' + str(i)  + ' PCA components is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm accuracy for ' + str(i) + ' PCA components is ' + str(accuracy_score(number_labels,y_train)))
    start = time.time()
    centers, labels = improved_kmeans(X_pca, num_clusters)
    end = time.time()
    print('improved Kmeans algorithm running time for ' + str(i)  + ' PCA components is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_train)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
                number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm accuracy for ' + str(i)  + ' PCA components is ' + str(accuracy_score(number_labels,y_train)))
        
#Iris dataset
iris = pd.read_csv(Path/To/Iris/Dataset/IRIS.csv')
y_iris = iris.iloc[:, 4].values
y_iris[y_iris == 'Iris-setosa'] = 0
y_iris[y_iris == 'Iris-versicolor'] = 1
y_iris[y_iris == 'Iris-virginica'] = 2
y_iris = np.array(y_iris).astype('int64')
X_iris = iris.iloc[:,0:4].values


num_centers = [3,4,5,6]
for i in (num_centers):
    start = time.time()
    centers, labels = K_means(X_iris, i)
    end = time.time()
    print('my original Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_iris)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_iris)))
    start = time.time()
    centers, labels = improved_kmeans(X_iris, i)
    end = time.time()
    print('improved Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_iris)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_iris)))
    start = time.time()
    centers_init = initial_centroids_max_var(X_iris,i)
    centers, labels = K_means(X_iris, i,centers_init)
    end = time.time()
    print('my original Kmeans algorithm with function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_iris)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_iris)))
    start = time.time()
    centers_init = initial_centroids_max_var(X_iris,i)
    centers, labels = improved_kmeans(X_iris, i,centers_init)
    end = time.time()
    print('improved Kmeans algorithm with function for centroid initialization running time for ' + str(i)  + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,y_iris)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,y_iris)))

data = pd.read_csv(Path/To/Letter/Dataset/letter-recognition.csv')
data = data.fillna(value = -1)
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for i,letter in enumerate (letters):
    data['letter'] = data['letter'].replace(letter, i+1)
#Letter dataset
X = data.drop('letter',axis=1)
Y =data['letter']
X = X.to_numpy()
#Y = Y.to_numpy()
num_centers = [26,50,75,90]
for i in (num_centers):
    start = time.time()
    centers, labels = K_means(X, i)
    end = time.time()
    print('my original Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
    start = time.time()
    centers, labels = improved_kmeans(X, i)
    end = time.time()
    print('improved Kmeans algorithm with random centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with random centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
    start = time.time()
    centers_init = initial_centroids_x(X,i)
    centers, labels = K_means(X, i,centers_init)
    end = time.time()
    print('my original Kmeans algorithm with function for centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
    start = time.time()
    centers_init = initial_centroids_x(X,i)
    centers, labels = improved_kmeans(X, i,centers_init)
    end = time.time()
    print('improved Kmeans algorithm with function for centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with function for centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
    start = time.time()
    centers_init = initial_centroids_max_var(X,i)
    centers, labels = K_means(X, i,centers_init)
    end = time.time()
    print('my original Kmeans algorithm with max var centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('my original Kmeans algorithm with max var centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
    start = time.time()
    centers_init = initial_centroids_max_var(X,i)
    centers, labels = improved_kmeans(X, i,centers_init)
    end = time.time()
    print('improved Kmeans algorithm with max var centroid initialization running time for ' + str(i) + ' centers is ' + str( - start + end) + ' sec')
    reference_labels = retrieve_info(labels,Y)
    number_labels = np.random.rand(len(labels))
    for j in range(len(labels)):
            number_labels[j] = reference_labels[labels[j]]
    print('improved Kmeans algorithm with max var centroid initialization accuracy for ' + str(i) + ' centers is ' + str(accuracy_score(number_labels,Y)))
