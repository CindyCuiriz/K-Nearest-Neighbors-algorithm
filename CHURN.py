# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:54:40 2023

@author: jocelyn
"""

# MACHINE LEARNING 
# ALGORITHM: K-Nearest Neighbors
#KNN MODEL
# CHURN DATASET

#Importar librerías

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#from scipy import stats
from sklearn.model_selection import train_test_split

'''
to ignore warnings temporarely
'''
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


'Dirección'
direc = 'C:\\Users\\jocelyn\\Desktop\\Python\\MachineLearning\\'

'Creamos nuestro Dataframe'
churn_df = pd.read_csv(direc + "churn.csv", index_col="index")
churn_df.drop("Unnamed: 8", axis=1, inplace=True)

'''
One excercise taking into consideration total_day_charge and total_eve_charge
#X, a 2D array of our features
X = churn_df[['total_day_charge','total_eve_charge']].values
#y, a 1D array of the target values - in this case, churn status
y = churn_df['churn'].values

print(X.shape,y.shape)

#KNeighborsClassifier, setting n_neighbors equal to 15
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X,y)

#Predicting on unlabeled data

X_new = np.array([[56.8,17.5],[24.4,24.1],[50.1,10.9]])
print(X_new.shape)
#3 observations, 2 features

predictions=knn.predict(X_new)
print("Predictions: {}".format(predictions))
'''


'''
#This is for FutureWarning
neigh_ind = knn.kneighbors(X_new, return_distance=False)
y_neighbors = y[neigh_ind]
predictions, _ = stats.mode(y_neighbors, axis=1, keepdims=True)
print("Predictions: {}".format(predictions.ravel()))

'''

# Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values
# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
# Fit the classifier to the data
knn.fit(X, y)
X_new = np.array([[10.0, 50.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
# Predict the labels for the X_new
y_pred = knn.predict(X_new)
# Print the predictions for X_new
print("Predictions: {}".format(y_pred)) 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))



'''
this is for Train/test split + computing accuracy
'''

#churn_df.drop("churn",axis=1,inplace=True)
X = churn_df.values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

'''
Interpreting model complexity overfitting vs underfitting
'''
# Create neighbors
neighbors = np.arange(1,13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train,y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train,y_train)
	test_accuracies[neighbor] = knn.score(X_test,y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)


'''
Visualizing model complexity
'''

# Plot training accuracies
plt.plot(neighbors,train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors,test_accuracies.values(), label="Testing Accuracy")

# Add a title
plt.title("KNN: Varying Number of Neighbors")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

'''
RESULTS
testing accuracy peaks with 2  neighbors, which would be the optimat value for our model
Training accuracy and Testing accuracy decrease as the number of neighbors gets larger
This might tell us that our model has not that good information to work with, mainly because 
it is random data I created to test KNN models! But when applied to better information
This model will help us create better results

'''
