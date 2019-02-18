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
import matplotlib.pyplot as plt


from pandas import read_csv

from sklearn import preprocessing
from sklearn import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def main():
    ########################### Variable dictionary #########################
    # featureDataSet is a pandas DataFrame that contains all the data for working
    # foldNumber is number of folds, data is divided to "foldNumber" folds 
    #featureNumber is maximum number of features that are used
    # maxNeighbours is maximum number of neighbours used in KNN
    # scoreMatrix is a matrix, rows are number of features and colummn number of neighbours. elemts are scores
    #########################################################################
    
    # read the file
    df= read_csv("A2_t2_dataset.tsv",sep="\t",header=None)
    
    #shuffle the rows
    df=utils.shuffle(df).reset_index(drop=True)

    featureDataSet= df.iloc[:,:-1] # select only the features
    target= df.iloc[:,-1:] # class column
    P= featureDataSet.shape[1] # number of features
 
    # Unsupervised feature selection and preparation
    featureDataSet=Feature_Sort(featureDataSet)
    featureDataSet=Feature_Select_Corr(featureDataSet)
    #featureDataSet=Feature_Scaling(featureDataSet)
 
    
    #Setting performance variables:
    TotalfoldNumber = 5 # number of folds
    featureNumber = 8 # maximum number of features to be used 
    maxNeighbours = 8 # maximum number of neighbours used in KNN
    # scoreMAtrix keeps scores for different KNN and features selection
    scoreMatrix= pd.DataFrame(np.zeros((featureNumber,maxNeighbours)))
    
    print("Calculation has started, please wait. We will keep you updated")

    
    # Feature selection 
    for f in range(1, featureNumber+1):
        
        
        # f+1 is the number of selected features to work with in the f'th Iteration
        # features_working is the first i'th feature selected to work with
        features_working = featureDataSet.iloc[:,:f]
        
        # KNN implementation
        for neigborNumber in range(1,maxNeighbours+1):
            #k is number of neighbours for each itteration
            
            
            # Mergging features and target to fold them
            totalDF= pd.concat((features_working,target),axis=1) # a complete sebset of data
            totalDF.columns=range(totalDF.shape[1])
           # print(totalDF)
            
           
           # folds is a 3D matrix with a shape of = {totalfolds* rows per fold* (feature+target)}
            folds = SplitFold(totalDF,TotalfoldNumber)
            
            for selectedFold in range(TotalfoldNumber):
                
                # converts folds to test and train set
                dfTrain , dfTest = Mergefolds(folds,selectedFold,TotalfoldNumber)
                #print(f,k,selectedFold)
                
                # Make datasets ready for KNN
                X_train = dfTrain.iloc[:,:-1] # all but the last column
                X_test = dfTest.iloc[:,:-1]

                y_train=dfTrain.iloc[:,-1:].squeeze() # the target column
                y_test=dfTest.iloc[:,-1:].squeeze()
                
                
                # implement KNN for the selectedFold:
                neighbor = KNeighborsClassifier(n_neighbors=neigborNumber)
                neighbor.fit(X_train, y_train)
                y_predicted = neighbor.predict(X_test)
                accuracy = neighbor.score(X_test, y_test) # if need only KNN accuracy un comment this
                #accuracy = roc_auc_score(y_test, y_predicted) # Area under the curve by sklearn
#                print(accuracy)
#
#                print(f,neigborNumber)
               
                #store the cummulativescore of KNN for each fold in 
                scoreMatrix.iloc[f-1,neigborNumber-1] = scoreMatrix.iloc[f-1,neigborNumber-1] + accuracy
                # neigborNumber-1 : hint: reason for -1 is that the loop starts from 1 and goes to
                # neigborNumber+1 (Because KNN does not accept 0)
        print("{:.2f} % is done.".format((f)/featureNumber*100))
                
    # take the averge of scores
    scoreMatrix = scoreMatrix.dropna() / TotalfoldNumber
    print(scoreMatrix)
    maxScore=0.0
    for i in range(featureNumber-1):
        for j in range(maxNeighbours-1):
            if (scoreMatrix.iloc[i,j] > maxScore):
                bestNumberofFeature = i+1
                bestNumberofNeighbours = j+1
                maxScore=scoreMatrix.iloc[i,j]
    
    print("The best number of features is {}, The best number of neighbors is {}".format(bestNumberofFeature,bestNumberofNeighbours))
    print("The accuracy for aforementioned values is: {0:.4f}".format(maxScore))      
    SurfacePlot(scoreMatrix)      
                
    
    
                       
                
def Mergefolds(folds,selectedFold,TotalfoldNumber):
    
    foldColumn = folds.shape[2] # number of column for this fold
    foldRow = folds.shape[1] # number of rows for this fold

    # Testset is the selected fold, drop possible NaN and reset its index
    dfTest = pd.DataFrame(folds[selectedFold,:,:]).dropna().reset_index(drop=True)
    
    # To form Trainset: 1- delete selected fold
    dfTrain= (np.delete(folds, selectedFold, axis=0))
    # 2- Reshape it to proper size
    dfTrain= pd.DataFrame(dfTrain.ravel().reshape((TotalfoldNumber-1) * foldRow,foldColumn))
    # 3- drop NaN and reset the index
    dfTrain= dfTrain.dropna().reset_index(drop=True)

    return  dfTrain, dfTest



def SplitFold(totalDF,TotalfoldNumber):

    rows = totalDF.shape[0] # number of total rows 
    foldRow = math.ceil(rows/TotalfoldNumber) # maximum number of rows in each fold  
    foldColumn = totalDF.shape[1] # number of features 
    folds= np.empty((TotalfoldNumber,foldRow,foldColumn)) # creating empty folds, 
    folds[:]=np.nan # set the fold toNan so eliminating them would be easy by pd.dropna
    
    # if need to shuffle use this:
    # suffle them with sklearn 
    # from sklearn import utils
    # initial=utils.shuffle(initial)
    
    classA= totalDF.loc[totalDF[foldColumn-1]==0] # the one with 0 in the last column are class A
    classB= totalDF.loc[totalDF[foldColumn-1]==1]

    cA=classA.values # convert pandas Data frame to numpy array
    cB=classB.values
    sizeA=cA.shape[0] # sizeA is the number of instances in class A
    
    for i in range(sizeA): # asigning almost equal number of class A and B to folds
            folds[i%TotalfoldNumber,i//TotalfoldNumber,:]=cA[i,:]

    for i in range(sizeA,rows,1): # we start from sizeA to continue with the next fold, not necceserily the first one
            folds[i%TotalfoldNumber,i//TotalfoldNumber,:]=cB[i-sizeA,:]
    # Untill here we read the data, devided into k fold, each fold has same number of class A and B
    return folds
 
        

            
            
        
    
    
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
                featureDataSet.iloc[:,i]= np.nan   # ignoring the second one by setting to NaN
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
    sc=StandardScaler()
    print("Features are scaled")
    return pd.DataFrame(sc.fit_transform(featureDataSet.astype(float)))


def SurfacePlot(scoreMatrix):

    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X = np.arange(scoreMatrix.shape[0])
    Y = np.arange(scoreMatrix.shape[1])
    X, Y = np.meshgrid(X, Y)

    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.array(scoreMatrix), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0.88, 0.915)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Number of neighbors')
    ax.set_zlabel('KNN accuracy')
    ax.set_title("Surface plot for accuracy")
    plt.show()

    
if __name__ == "__main__":main() 
