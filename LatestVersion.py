# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:13:28 2019

@author: Javad
"""
import math
import time
import pandas as pd
import numpy as np
import sklearn as sk

from pandas import read_csv

from sklearn import preprocessing
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


def main():
    ########################### Variable dictionary #########################
    # featureDataSet is a pandas DataFrame that contains all the data for working
    # foldNumber is number of folds, data is divided to "foldNumber" folds
    # featureNumber is maximum number of features that are used
    # maxNeighbours is maximum number of neighbours used in KNN
    # scoreMatrix is a matrix, rows are number of features and colummn number of neighbours. elemts are scores
    #########################################################################

    # read the file
    df = read_csv("A2_t2_dataset.tsv", sep="\t", header=None)

    # shuffle the rows
    df = utils.shuffle(df).reset_index(drop=True)

    featureDataSet = df.iloc[:, :-1]  # select only the features
    target = df.iloc[:, -1:]  # class column
    P = featureDataSet.shape[1]  # number of features

    # Unsupervised feature selection and preparation
    featureDataSet=Feature_Select_Corr(featureDataSet)
    featureDataSet=Feature_Sort(featureDataSet)
    featureDataSet=Feature_Scaling(featureDataSet)
 
    
    #Setting performance variables:
    TotalfoldNumber = 5 # number of folds
    featureNumber = 8 # maximum number of features to be used 
    maxNeighbours = 12 # maximum number of neighbours used in KNN
    # scoreMAtrix keeps scores for different KNN and features selection
    scoreMatrix = pd.DataFrame(np.zeros((featureNumber, maxNeighbours)))

    print("Calculation has started, please wait. We will keep you updated")

    # Feature selection
    for f in range(featureNumber):

        # f+1 is the number of selected features to work with in the f'th Iteration
        # features_working is the first i'th feature selected to work with
        features_working = featureDataSet.iloc[:, :f + 1]

        # KNN implementation
        for neigborNumber in range(1, maxNeighbours + 1):
            # k is number of neighbours for each itteration

            # Mergging features and target to fold them
            # a complete sebset of data
            totalDF = pd.concat((features_working, target), axis=1)
            totalDF.columns = range(totalDF.shape[1])
           # print(totalDF)

           # folds is a 3D matrix with a shape of = {totalfolds* rows per fold*
           # (feature+target)}
            folds = SplitFold(totalDF, TotalfoldNumber)

            for selectedFold in range(TotalfoldNumber):

                # converts folds to test and train set
                dfTrain, dfTest = Mergefolds(
                    folds, selectedFold, TotalfoldNumber)
                # print(f,k,selectedFold)

                # Make datasets ready for KNN
                X_train = dfTrain.iloc[:, :-1]  # all but the last column
                X_test = dfTest.iloc[:, :-1]

                y_train = dfTrain.iloc[:, -1:].squeeze()  # the target column
                y_test = dfTest.iloc[:, -1:].squeeze()

                # implement KNN for the selectedFold:
                neighbor = KNeighborsClassifier(n_neighbors=neigborNumber)
                neighbor.fit(X_train, y_train)
                y_predicted = neighbor.predict(X_test)
                # if need only KNN accuracy un comment this
                accuracy = neighbor.score(X_test, y_test)
                # accuracy = roc_auc_score(y_test, y_predicted) # Area under the curve by sklearn
#                print(accuracy)
#
#                print(f,neigborNumber)

                # store the cummulativescore of KNN for each fold in
                scoreMatrix.iloc[f, neigborNumber - 1] += accuracy
                # neigborNumber-1 : hint: reason for -1 is that the loop starts from 1 and goes to
                # neigborNumber+1 (Because KNN does not accept 0)
        print("{:.2f} % is done.".format((f + 1) / featureNumber * 100))

    # take the averge of scores
    scoreMatrix = scoreMatrix / TotalfoldNumber
    print(scoreMatrix)
    maxScore = 0.0
    for i in range(featureNumber):
        for j in range(maxNeighbours - 1):
            if (scoreMatrix.iloc[i, j] > maxScore):
                bestNumberofFeature = i + 1
                bestNumberofNeighbours = j + 1
                maxScore = scoreMatrix.iloc[i, j]

    print(
        "The best number of features is {}, The best number of neighbors is {}".format(
            bestNumberofFeature,
            bestNumberofNeighbours))
    print(
        "The accuracy for aforementioned values is: {0:.4f}".format(maxScore))


def Mergefolds(folds, selectedFold, TotalfoldNumber):

    foldColumn = folds.shape[2]  # number of column for this fold
    foldRow = folds.shape[1]  # number of rows for this fold

    # Testset is the selected fold, drop possible NaN and reset its index
    dfTest = pd.DataFrame(folds[selectedFold, :, :]
                          ).dropna().reset_index(drop=True)

    # To form Trainset: 1- delete selected fold
    dfTrain = (np.delete(folds, selectedFold, axis=0))
    # 2- Reshape it to proper size
    dfTrain = pd.DataFrame(
        dfTrain.ravel().reshape(
            (TotalfoldNumber - 1) * foldRow,
            foldColumn))
    # 3- drop NaN and reset the index
    dfTrain = dfTrain.dropna().reset_index(drop=True)

    return dfTrain, dfTest


def SplitFold(totalDF, TotalfoldNumber):

    rows = totalDF.shape[0]  # number of total rows
    # maximum number of rows in each fold
    foldRow = math.ceil(rows / TotalfoldNumber)
    foldColumn = totalDF.shape[1]  # number of features
    folds = np.empty((TotalfoldNumber, foldRow, foldColumn)
                     )  # creating empty folds,
    # set the fold toNan so eliminating them would be easy by pd.dropna
    folds[:] = np.nan

    # if need to shuffle use this:
    # suffle them with sklearn
    # from sklearn import utils
    # initial=utils.shuffle(initial)

    # the one with 0 in the last column are class A
    classA = totalDF.loc[totalDF[foldColumn - 1] == 0]
    classB = totalDF.loc[totalDF[foldColumn - 1] == 1]

    cA = classA.values  # convert pandas Data frame to numpy array
    cB = classB.values
    sizeA = cA.shape[0]  # sizeA is the number of instances in class A

    for i in range(sizeA):  # asigning almost equal number of class A and B to folds
        folds[i % TotalfoldNumber, i // TotalfoldNumber, :] = cA[i, :]

    for i in range(sizeA, rows, 1):  # we start from sizeA to continue with the next fold, not necceserily the first one
        folds[i % TotalfoldNumber, i // TotalfoldNumber, :] = cB[i - sizeA, :]
    # Untill here we read the data, devided into k fold, each fold has same
    # number of class A and B
    return folds


def Feature_Select_Corr(featureDataSet):
    """ Eliminate some of the features that are correlated and keep one"""
    print("Start the feature selection based on correlation")
    start = time.time()  # let's time each step
    # Unsupervised feature selection
    # based on the correlation between features
    # this is the correlation between all features so it is a P*P matrix
    # this is the correlation between all features so it is a P*P matrix
    cordf = featureDataSet.corr()
    # Going through all features' correlation and set the features with high correlation( >0.5) to NaN but one of them
    # By setting them to NaN, we mark them and later we can easily drop them
    P = featureDataSet.shape[1]  # number of features

    for j in range(P - 1):
        for i in range(
                j + 1,
                P):  # moving in the upper triangle of the correlation matrix
            if (abs(cordf.iloc[j, i]) > 0.5):
                featureDataSet.iloc[:, i] = np.nan
    working_df = featureDataSet.copy()
    # dropping NaN, which are highly correlated features
    notCorrFeatures = working_df.dropna(axis=1)
    notCorrFeatures.columns = range(notCorrFeatures.shape[1])

    end = time.time()
    print("Feature selection based on correlation took {0:0.2f} seconds ".format(end - start))
    return notCorrFeatures


def Feature_Sort(featureDataSet):
    relStd = featureDataSet.std() / featureDataSet.mean()  # rel std calculation
    # Hint: (relStd.sort_values(ascending=False).index) # extracting the index
    # of relative std of the features
    print("Features are sorted based on relative standard deviation")
    # sorting features based on rel std by reindexing
    return featureDataSet.reindex(
        relStd.sort_values(
            ascending=False).index, axis=1)


def Feature_Scaling(featureDataSet):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    print("Features are scaled")
    return pd.DataFrame(sc.fit_transform(featureDataSet.astype(float)))


if __name__ == "__main__":
    main()
