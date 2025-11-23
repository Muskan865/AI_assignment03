"""
CS 351 - Artificial Intelligence 
Assignment 3, Question 1

Student 1(Name and ID): Muskan Rehan (mr09207)
Student 2(Name and ID): Aliza Fatima (af09188)

"""
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KMeansClustering:
    
    def __init__(self, filename: str, K:int):
        self.image  = mpimg.imread(filename)
        self.original_image = self.image.copy()
        self.K = K
        self.pixels = self.image.reshape(-1, 3)
        self.centroids_history = []
        self.sse_history = []
        
    
    def __generate_initial_centroids(self) ->list :
        #write your code here to return initial random centroids
        idx = np.random.choice(len(self.pixels), self.K, replace=False)
        return [tuple(self.pixels[i]) for i in idx]
    

    def __calculate_distance(self, p1: tuple, p2: tuple) -> float:
        #This function computes and returns distances between two data points
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))


    def __assign_clusters(self)->dict:
        #assign each data point to its nearest cluster (centroid)
        clusters = {i: [] for i in range(self.K)}

        for idx, pixel in enumerate(self.pixels):
            distances = [self.__calculate_distance(pixel, c) for c in self.centroids]
            cluster_index = int(np.argmin(distances))
            clusters[cluster_index].append(idx)

        return clusters

    def __recompute_centroids(self)->list:
        #your code here to return new centroids based on cluster formation
        new_centroids = []
        for i in range(self.K):
            if len(self.clusters[i]) == 0:
                new_centroids.append(tuple(self.pixels[np.random.randint(len(self.pixels))]))
            else:
                new_centroids.append(tuple(np.mean(self.pixels[self.clusters[i]], axis=0)))
        return new_centroids


    def apply(self):
        #your code here to apply kmeans algorithm to cluster data loaded from the image file.
        self.centroids = self.__generate_initial_centroids()
        self.centroids_history.append(self.centroids)

        iteration = 0

        while True:
            iteration += 1
            self.clusters = self.__assign_clusters()
            new_centroids = self.__recompute_centroids()

            sse = 0
            for i in range(self.K):
                for p in self.clusters[i]:
                    sse += np.sum((self.pixels[p] - np.array(new_centroids[i]))**2)
            self.sse_history.append(sse)
            
            if np.allclose(self.centroids, new_centroids, atol=1e-2):
                break
            
            self.centroids = new_centroids
            self.centroids_history.append(self.centroids)

            if iteration < 7 or iteration % 5 == 0:
                self.__save_image()   
                plt.figure()
                plt.title(f"Iteration {iteration}")
                plt.imshow(self.segmented_image)
                plt.axis("off")
                plt.savefig(f"iteration_{iteration}.png")
                plt.close()
        self.__save_image()

    def __save_image(self):
        #This function overwrites original image with segmented image to be shown later.
        new_img = np.zeros_like(self.pixels)

        for i in range(self.K):
            centroid = np.array(self.centroids[i])
            for idx in self.clusters[i]:
                new_img[idx] = centroid

        self.segmented_image = new_img.reshape(self.image.shape)


    def show_result(self):
        plt.figure()
        plt.imshow(self.segmented_image)
        plt.title(f"Clustered Image (K={self.K})")
        plt.axis("off")
        plt.show()


    def print_centroids(self):
        #This function prints all centroids formed by Kmeans clustering
        print("Centroids at each iteration:")
        for i, centroids in enumerate(self.centroids_history, 1):
            print(f"Iteration {i}:")
            for c in centroids:
                print(c)
            print("-" * 30)


    def quality(self):
        #This function computes the quality of clustering using the Sum of Squared Error (SSE) as a measure, 
        # and plots a graph to show how SSE changes during the clustering process.
        plt.plot(self.sse_history)
        plt.title("SSE vs Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.grid(True)
        plt.show()



kmeans = KMeansClustering("images\\sample1.jpg", K=5)
kmeans.apply()
kmeans.show_result()
kmeans.print_centroids()
kmeans.quality()
        