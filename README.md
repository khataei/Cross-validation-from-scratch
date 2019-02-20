# Cross-validation from scratch: Feature selection, Tunning model parameters

### Summary:
Here I import a dataset, preprocess the data and then try to find the best combination of features and KNN neighbor numbers.
### Goal:
The goal of this code is to implement the cross-validation method to find the most important features and the best parameter of a model.  
First is preparing the data, i.e., dealing with missing values,  sorting them based on their relative standard deviation,  and scaling the data. Next step is explained by this pseudo code:

## Pseudo-code of the whole program:
Let F total number of selected features
Let K total number of folds
Let N total number of neighbors for KNN implementation
1.    Prepare the data
a.    Read the data and variables
b.    Shuffle rows
c.    Split the data to features and target datasets
d.    Sort the features based on importance, i.e. relative standard deviation
e.    Eliminate features based on their correlations, i.e. Unsupervised feature selection
f.    Scale features
2.    For f from 1 to F
a.    For n from 1 to N
i.    Split the data pseudo randomly into K folds
ii.    For k from 1 to K
1.    Set the k’th fold as test dataset and merge all other folds and set to train dataset
2.    Implement KNN with n neighbor
3.    Store the KNN accuracy in a 2D matrix
3.    Calculated the average score by dividing the matrix by K
4.    Find the maximum score and return its indices as best number of features and neighbors

##Pseudo code for Cross Validation:
### Split Part:
1-    Create an empty 3-D array to store folds
2-    Divide the data based on the class in to 2 classes, A and B
a.    For s from 1 to size of class A:
i.    For k from 1 to number of folds
1.    Send Class A instances, s,  to k’th fold
b.    For s from 1  to size of class B 
1.    Send Class B instances, s,  to k’th fold

### Merge part:
1-    Merge all the folds but the selected one
2-    Drop all NaN values
3-    Return the selected fold as the test set and the merged folds as the training set
Loss function:
To compare performances between different selections of features and KNN parameters, we have used the accuracy given by KNN. We implemented Scikit-Learn function and used its score. The scores for all folds are added, and then the average is calculated and considered for comparison. All the accuracies are stored in a matrix and can be used to draw 2-D and 3-D plots.







