# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:13:28 2019

@author: Javad
"""
import pandas as pd
import numpy as np
import time
import sklearn as sk
from sklearn import preprocessing


def main():
    from pandas import read_csv
    ########################### Variable dictionary #########################
    # featureDataSet is a pandas DataFrame that contains all the data for working
    # foldNumber is number of folds, data is divided to "foldNumber" folds 
    #featureNumber is maximum number of features that are used
    # maxNeighbours is maximum number of neighbours used in KNN
    #scoreMatrix is a matrix, rows are number of features and colummn number of neighbours. elemts are scores
    #########################################################################
    
    # read the file
    df= read_csv("A2_t2_dataset.tsv",sep="\t",header=None)
    featureDataSet= df.iloc[:,:-1] # select only the features
    target=df.iloc[:,-1:] # class column
    P= featureDataSet.shape[1] # number of features
 
    # Unsupervised feature selection and preparation
    featureDataSet=Feature_Select_Corr(featureDataSet)
    featureDataSet=Feature_Sort(featureDataSet)
    featureDataSet=Feature_Scaling(featureDataSet)
 
    
    #Setting performance variables:
    foldNumber=5 # number of folds
    featureNumber=21 # maximum number of features to be used 
    maxNeighbours=21 # maximum number of neighbours used in KNN
    # scoreMAtrix keeps scores for different KNN and features selection
    scoreMatrix= pd.DataFrame(np.zeros((featureNumber,maxNeighbours)))
    
    
    # Feature selection 
    for f in range(featureNumber):
        # f+1 is the number of selected features to work with in the f'th Iteration
        # features_working is the first i'th feature selected to work with
        features_working= featureDataSet.iloc[:,:f+1]
        
        # KNN implementation
        for k in range(1,maxNeighbours+1):
            #k is number of neighbours for each itteration
            
            # Calling the folding function implemented from scratch
            
        
    
    
def Feature_Select_Corr(featureDataSet):
    """ Eliminate some of the features that are correlated and keep one"""
    print("Start the feature selection based on correlation")
    start = time.time() # let's time each step
    # Unsupervised feature selection
    # based on the correlation between features
    # this is the correlation between all features so it is a P*P matrix
    cordf= featureDataSet.corr() # this is the correlation between all features so it is a P*P matrix
    #Going through all features' correlation and set the features with high correlation( >0.5) to NaN but one of them
    # By setting them to NaN, we mark them and later we can easily drop them
    P= featureDataSet.shape[1] # number of features
    
    for j in range(P-1):
        for i in range(j+1,P): # moving in the upper triangle of the correlation matrix
            if (abs(cordf.iloc[j,i]) > 0.5):
                featureDataSet.iloc[:,i]= np.nan   
    working_df= featureDataSet.copy()
    notCorrFeatures=working_df.dropna(axis=1) # dropping NaN, which are highly correlated features
    notCorrFeatures.columns = range(notCorrFeatures.shape[1])
    
    end = time. time()
    print("Feature selection based on correlation took {0:0.2f} seconds ".format(end-start))
    return notCorrFeatures


def Feature_Sort(featureDataSet):
    relStd=featureDataSet.std()/featureDataSet.mean() # rel std calculation
    #Hint: (relStd.sort_values(ascending=False).index) # extracting the index of relative std of the features
    print("Features are sorted based on relative standard deviation")
    # sorting features based on rel std by reindexing
    return featureDataSet.reindex(relStd.sort_values(ascending=False).index,axis=1)


def Feature_Scaling(featureDataSet):
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    print("Features are scaled")
    return pd.DataFrame(sc.fit_transform(featureDataSet.astype(float)))

    
if __name__ == "__main__":main() 
