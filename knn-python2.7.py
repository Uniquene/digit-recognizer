


import os.path
import csv 
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

data_dir = "/home/dmcl124/Documents/JuneBooks/KaggleDigital_DS"


def opencsv():
    data = pd.read_csv(os.path.join(data_dir,'train.csv'))
    data1 = pd.read_csv(os.path.join(data_dir,'test.csv'))
    
    print type(data)
    
    train_data = data.values[0:,1:]
    train_label = data.values[0:,0]
    test_data = data1.values[0:,0:]
    return train_data,train_label,test_data

opencsv()



def saveResult(result,csvName):
    with open(csvName,'w') as myfile:
        myWriter = csv.writer(myfile)
        myWriter.writerow(['ImageId','Lable'])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index,int(r)])
    print 'Saved successfully...'

def knnClassify(trainData,trainLabel):
    knnClf = KNeighborsClassifier()
    knnClf.fit(trainData,np.ravel(trainLabel))
    return knnClf

def dRPCA(x_train,x_test,COMPONENT_NUM):
    print 'dimensionality reduction...'
    trainData = np.array(x_train)
    testData = np.array(x_test)
    pca = PCA(n_components = COMPONENT_NUM,whiten = False)
    pca.fit(trainData)
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)
    
    # 属性：
    # - components_ :主成分组数
    # - explained_variance_ratio_:每个主成分占方差比例
    # - n_components_ :一个整数，指示主成分有多少个元素
    
    print pca.explained_variance_, '\n', pca.explained_variance_ratio_,'\n',pca.n_components_
    print sum(pca.explained_variance_ratio_)
    return pcaTrainData,pcaTestData

def dRecognition_knn():
    start_time = time.time()
    
    trainData,trainLabel,testData = opencsv()
    print "tainData==>",type(trainData),np.shape(trainData)
    print "trainLabel==>",type(trainLabel),np.shape(trainLabel)
    print "testData==>",type(testData),np.shape(testData)
    print "load data finish"
    stop_time_1 = time.time()
    print "load data time used:%f" % (stop_time_1-start_time)
    
    trainData,testData = dRPCA(trainData,testData,0.8)
    
    knnClf = knnClassify(trainData,trainLabel)
    
    testLabel = knnClf.predict(testData)
    
    saveResult(testLabel,os.path.join(data_dir,'Result_sklearn_knn.csv'))
    print "finish!"
    stop_time_r = time.time()
    print "classify time used:%f" % (stop_time_r - start_time)
    
    

if __name__ == '__main__':
    dRecognition_knn()

