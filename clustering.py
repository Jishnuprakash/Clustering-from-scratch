#%%
#---------------------------------------------------------------------
# Import required Libraries and Read data
#---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Load train and test data
def readData(data):
    """
    Reads data and converts to numpy arrays 
    """
    dat = np.array([i.split(',') for i in open(data,"r").read().splitlines()])
    dat = dat[1:].astype(float)
    return dat

#Read data
iris = readData("iris.csv")

#Merge to single array
data = iris[:, 0:4]
#Create labels for each data item
labels = iris[:, 4]
#%%
#---------------------------------------------------------------------
# Define class and functions for implementing K-means and K-medians,
# Measuring B-CUBED precision, recall and F1-score,
# and plotting the scores
#---------------------------------------------------------------------
class kRepresentative():
    """
    K-Representative Clustering
    """

    def __init__(self, k, data, kmean):
        """
        k: Number of clusters
        data: input data
        kmean: True for K-means algorithm, False for K-medians
        """
        self.k = k
        self.kmean = kmean
        self.data = data

    def distance(self, x,y):
        """
        Calculates distance between x and y
        Squared Euclidean distance for K-means
        Manhattan distance for K-medians
        """
        if self.kmean:
            #Squared Euclidean distance
            out = np.linalg.norm(x-y)**2
        else:
            #Manhattan distance
            out = np.linalg.norm(x-y, ord=1)
        return out

    def genCentroids(self):
        """
        Generates centroids randomly from data
        """
        #Set seed value to reproduce same results
        np.random.seed(7)
        indices = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = {}
        # Choosing random objects from data as centroids
        for i in range(self.k):
            self.centroids[i+1] = self.data[indices[i]]

    def assignCentroid(self):
        """
        Assigns centroids to each data point and form clusters
        """
        #Initialise clusters
        self.clusters = {i:[] for i in self.centroids.keys()}
        for i in self.data:
            #Compute distances
            distances = [self.distance(i, k) for k in self.centroids.values()]
            cluster = distances.index(min(distances)) + 1
            #Assign cluster
            self.clusters[cluster].append(i)
    
    def objFunction(self, centroids):
        """
        Computes objective function using respective distance
        Sum of Squared error(SSE) for k-means
        Sum of L1 (Manahattan) distance for k-medians
        """
        obj = 0
        for i in self.clusters:
            distances = [self.distance(j, centroids[i]) for j in self.clusters[i]]
            obj += sum(distances)
        return obj
    
    def fit(self, epochs):
        """
        Training of K-Representative algorithm
        epochs: Number of iterations
        """
        #Initialisation phase
        ##Generate initial centroids
        self.genCentroids()

        #Assignment Phase
        ##Assign data points to each centroids
        self.assignCentroid()

        # Uncomment to see cluster population when Initialized
        # for key, value in self.clusters.items():
        #     print("Assignment Phase: \nCluster {} assigned to {} datapoints".format(key, len(value)))
        
        for i in range(epochs):

            #Optimisation and Assignment Phases
            #Compute and store error
            prevError = self.objFunction(self.centroids)

            for k in self.centroids:
                if self.kmean:
                    self.centroids[k] = np.mean(self.clusters[k], axis=0)
                else:
                    self.centroids[k] = np.median(self.clusters[k], axis=0)

            #Compute new error
            currError = self.objFunction(self.centroids)
            #Check if optimised
            optimized = False if currError < prevError else True

            if not optimized:
                self.assignCentroid()
                # Uncomment to see live cluster population
                # print("For epoch {}: {}".format(i+1, [len(i) for i in self.clusters.values()]))
            else:
                print("For k={}, Objective function Optimised in epoch {}".format(len(self.clusters), i))
                break
        

def measure(kRep, labels):
    """
    Compute Precision, Recall and F1-Score
    kRep: K-Representative object
    labels: True Labels
    """
    for cluster in kRep.clusters:
        precisions, recalls, fscores = [], [], []
        #Label each item in all clusters 
        clusterLabels = []
        for i, j in zip(labels, kRep.data):
            for k in kRep.clusters[cluster]:
                if np.array_equal(j, k):
                    clusterLabels.append(i)
        #Compute precision, recall and f1-score for each item
        for i in clusterLabels:    
            p = clusterLabels.count(i)/ len(clusterLabels)
            r = clusterLabels.count(i)/ list(labels).count(i)
            f = (2*p*r) / (p+r)
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
    precision = round(np.mean(precisions), 2)
    recall = round(np.mean(recalls), 2)
    f1score = round(np.mean(fscores), 2)
    return precision, recall, f1score

def validate(epochs, data, kmean):
    """
    Intialise K-Representative clustering and compute Precision, recall and f1-score
    """
    precision, recall, f1score = [], [], []
    for k in range(1,10):
        kmeans = kRepresentative(k=k, data=data, kmean=kmean)
        kmeans.fit(epochs)
        valid = measure(kmeans, labels)
        precision.append(valid[0])
        recall.append(valid[1])
        f1score.append(valid[2])
    return precision, recall, f1score

def plotResults(precision, recall, f1score, ylabel, title):
    """
    Plot results together
    """
    # Plot the results
    plt.plot(range(1,10), precision, color='blue', marker='o', label='Precision')
    plt.plot(range(1,10), recall, color='red', marker='o', label='Recall')
    plt.plot(range(1,10), f1score, color='green', marker='o', label='F1-Score')
    plt.xlabel("Parameter k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

#%%
#---------------------------------------------------------------------
# K-Means 
#---------------------------------------------------------------------
#K-Means
print("\n K-Means \n\n")
precision, recall, f1score = validate(epochs=30, data=data, kmean=True)

# Plot the results
plotResults(precision, recall, f1score, "Evaluation of K-Means", 
                                        "K-Means evaluation metrics versus k")
                                        
#%%
#---------------------------------------------------------------------
# K-Medians 
#---------------------------------------------------------------------
#K-Medians
print("\n K-Medians \n\n")
precision, recall, f1score = validate(epochs=30, data=data, kmean=False)

# Plot the results
plotResults(precision, recall, f1score, "Evaluation of K-Medians", 
                                        "K-Medians evaluation metrics versus k")
#%%
#---------------------------------------------------------------------
# Best Performing Models
#---------------------------------------------------------------------
#K-Means with K=3
print("\n Comparing Best performing models \n\n")
kmeans_3 = kRepresentative(k=3, data=data, kmean=True)
kmeans_3.fit(30)
valid_kmeans = measure(kmeans_3, labels)

#K-Medians with K=3
kmedians_3 = kRepresentative(k=3, data=data, kmean=False)
kmedians_3.fit(30)
valid_kmedians = measure(kmedians_3, labels)

#Bar plot for comparison
plt.figure(figsize=(12,10))
ind = np.arange(3) 
width = 0.2  
plt.bar(ind, valid_kmeans, width, label='K-Means')
plt.bar(ind + width, valid_kmedians, width, label='K-Medians')
plt.ylabel('Scores')
plt.title('Comparison of K-means and K-medians clustering on data with K=3')
plt.xticks(ind + width / 2, ('Precision', 'Recall', 'F1-Score'))
plt.legend()
plt.show()

# %%
