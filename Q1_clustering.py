"""
CS 351 - Artificial Intelligence 
Assignment 3, Question 1

Student 1(Name and ID): Muskan Rehan (mr09207)
Student 2(Name and ID): Aliza Fatima (af09207)

"""
import numpy as np
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class KMeansClustering:
    
    def __init__(self, filename: str, K:int):
        self.image = mpimg.imread(filename)        
        self.K = K
    
    
    def __generate_initial_centroids(self) ->list :
        #write your code here to return initial random centroids
        pass
    

    def __calculate_distance(self, p1: tuple, p2: tuple) -> float:
        #This function computes and returns distances between two data points
        pass


    def __assign_clusters(self)->dict:
        #assign each data point to its nearest cluster (centroid)
        pass


    def __recompute_centroids(self)->list:
        #your code here to return new centroids based on cluster formation
        pass


    def apply(self):
        #your code here to apply kmeans algorithm to cluster data loaded from the image file.
        pass       
        

    def __save_image(self):
        #This function overwrites original image with segmented image to be shown later.
        pass


    def show_result(self):
        plt.imshow(self.image)


    def print_centroids(self):
        #This function prints all centroids formed by Kmeans clustering
        pass


    def quality(self):
        #This function computes the quality of clustering using the Sum of Squared Error (SSE) as a measure, 
        # and plots a graph to show how SSE changes during the clustering process.
        pass   
        
        
kmeans = KMeansClustering("images\sample1.jpg")
kmeans.apply()
kmeans.show_result()
kmeans.print_centroids()    
    
