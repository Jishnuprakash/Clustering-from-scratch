# Clustering-from-scratch
 K-Means and K-Medians from scratch using python

## About
K-Representative clustering (K-means and K-medians) is implemented in python from scartch.

## Requirements
python 3, numpy, matplotlib (pip install numpy matplotlib)

## Folder Structure
Please make sure, the files clustering.py and iris.csv are in the same location/folder. 

## How to run the code?
After installing requirements, it is very easy to run the clustering.py file. 
- Open the code in any Python IDE and click on run
or
- Open command prompt from project folder and run "python /clustering.py"

# Code in Detail
## Importing libraries
Importing numpy library for array operations and matplotlib for visualisation.

## Reading the data
readData Method is used to read iris.csv file and convert to numpy array.

## kRepresentative 
The class kRepresentative is used to implement k-means and k-medians algorithm from scratch.
 - initialised with k(user input), data and kmean (True or False)
 - squared Euclidean distance used for k-means and Manhattan for k-medians
 - Fit method is used for clustering
 
## Functions/Methods

- measure: used to compute B-Cubed precision, recall and f1-score
- validate: implement k-means and evaluate for k from 1 to 9
- plotResults: Plots measures to choose the best k value
