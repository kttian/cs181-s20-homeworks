# CS 181, Spring 2020
# Homework 4

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import distance
from seaborn import heatmap
import random as random

# This line loads the images for you. Don't change it! 
large_dataset = np.load("data/large_dataset.npy").astype(np.int64)
small_dataset = np.load("data/small_dataset.npy").astype(np.int64)
small_labels = np.load("data/small_dataset_labels.npy").astype(int)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that. 

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.mu = np.zeros(shape=(K,5)) #an array[K][D] of cluster centers
        self.z = np.zeros(shape=(5,K))  #an array[N][K] of cluster assignments
        self.obj = [] #list of (iter,objective)

    def loss(self,X):
        l = 0
        N = X.shape[0]
        K = self.K
        for n in range(N):
            for k in range(K):
                if(self.z[n][k]):
                    l += distance.euclidean(X[n],self.mu[k])
        return l

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        N = X.shape[0]
        D = X.shape[1]
        K = self.K
        self.mu = np.zeros(shape=(K,D))
        self.z = np.zeros(shape=(N,K))

        #random assign
        for i in range(N):
            k = random.randint(0,9)
            self.z[i][k] = 1

        count = 0
        max_iter = 100
        converged = False
        self.obj = []
        while(count <= max_iter and not converged):
            count += 1
            #update centers, for k in range(K):
            Nk = np.sum(self.z,axis=0)
            #print("Nk",Nk)
            #print(np.sum(Nk))
            self.mu = np.dot(np.transpose(self.z),X)
            # now divide each row k by Nk[k]
            for k in range(K):
                self.mu[k] = np.divide(self.mu[k], Nk[k])
            #print(X)
            #print(self.mu)

            #update assignment, for n in range(N):
            #self.z = np.zeros(shape=(N,K))
            converged = True
            for n in range(N):
                min_dist = distance.euclidean(X[n],self.mu[0])
                min_k = 0
                for k in range(1,K):
                    dk = distance.euclidean(X[n],self.mu[k])
                    if dk < min_dist:
                        min_dist = dk
                        min_k = k
                # set znk
                if self.z[n][min_k] != 1:
                    self.z[n] = np.zeros(K)
                    self.z[n][min_k] = 1
                    converged = False

            l = self.loss(X)
            self.obj.append((count,l))
        print("count", count)

    def plot_obj(self):
        px = [ob[0] for ob in self.obj]
        py = [ob[1] for ob in self.obj]
        plt.scatter(px,py)
        plt.show()   

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu

K = 10
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(large_dataset)
KMeansClassifier.plot_obj()
mean_images = KMeansClassifier.get_mean_images()

# This is how to plot an image. We ask that any images in your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(mean_images[0].reshape(28,28), cmap='Greys_r')
#plt.imshow(large_dataset[0].reshape(28,28), cmap='Greys_r')
plt.show()


class HAC(object):
	def __init__(self, linkage):
		self.linkage = linkage
        