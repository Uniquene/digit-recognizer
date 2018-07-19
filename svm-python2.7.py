#coding:utf-8

import os.path
import csv 
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt


data_dir = "/home/dmcl124/Documents/JuneBooks/KaggleDigital_DS"


def opencsv():
    print 'load Data...'
    data = pd.read_csv(os.path.join(data_dir,'train.csv'))
    dataPre = pd.read_csv(os.path.join(data_dir,'test.csv'))
    
    
    trainData = data.values[0:,1:]
    trainLabel = data.values[0:,0]
    preData = dataPre.values[0:,0:]
    return trainData,trainLabel,preData

opencsv()


def dRCsv(x_train,x_test,preData,COMPONENT_NUM):
    print 'dimensionality reduction...'
    trainData = np.array(x_train)
    testData = np.array(x_test)
    preData = np.array(preData)
    
    pca = PCA(n_components = COMPONENT_NUM,whiten = True)
    pca.fit(trainData)
    
    
    pcaTrainData = pca.transform(trainData)
    pcaTestData = pca.transform(testData)
    pcaPreData = pca.transform(preData)
    
    # 属性：
    # - components_ :主成分组数
    # - explained_variance_ratio_:每个主成分占方差比例
    # - n_components_ :一个整数，指示主成分有多少个元素
    
    print pca.explained_variance_, '\n', pca.explained_variance_ratio_,'\n',pca.n_components_
    print sum(pca.explained_variance_ratio_)
    return pcaTrainData,pcaTestData,pcaPreData


def trainModel(trainData,trainLabel):
    print 'Train SVM...'
    svmClf = SVC(C = 4, kernel = 'rbf') #惩罚参数
    svmClf.fit(trainData,trainLabel)
    return svmClf

def saveResult(result,csvName):
    with open(csvName,'w') as myfile:
        myWriter = csv.writer(myfile)
        myWriter.writerow(['ImageId','Lable'])
        index = 0
        for r in result:
            index += 1
            myWriter.writerow([index,int(r)])
    print 'Saved successfully...'

def analyse_data(dataMat):
    meanVals = np.mean(dataMat,axis = 0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved,rowvar = 0)
    
    eigvals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigvals)
    
    topNfeat = 100
    
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    sum_cov_score = 0
    cov_all_score = float(sum(eigvals))
    for i in range(0,len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        
        print '主成分：%s, 方差比：%s%%,累积方差占比：%s%%' % (format(i+1,'2.0f'), format(line_cov_score/cov_all_score * 100, '4.2f'),format(sum_cov_score/cov_all_score*100,'4.1f'))

def getOptimalAccuracy(trainData,trainLabel,preData):
    x_train,x_test,y_train,y_test = train_test_split(trainData,trainLabel,test_size=0.5)
    lineLen,featureLen = np.shape(x_test)
    
    minErr = 1
    minSumErr = 0
    optimalNum = 1 #最佳主成分的个数
    optimalLabel = []
    optiamlSVMClf = None  #最佳SVM模型
    pcaPreDataResult = None
    for i in range(30,45,1):
        pcaTrainData,pcaTestData,pcaPreData = dRCsv(x_train,x_test,preData,i)
        svmClf = trainModel(pcaTrainData,y_train)
        svmtestLabel = svmClf.predict(pcaTestData)
        
        errArr = np.mat(np.ones((lineLen,1)))
        sumErrArr = errArr[svmtestLabel!= y_test].sum()
        sumErr = sumErrArr/lineLen
        
        print 'i=%s' % i,lineLen,sumErrArr,sumErr
        if sumErr <= minErr:
            minErr = sumErr
            minSumErr = sumErrArr
            optimalNum = i
            optimalSVMClf = svmClf
            optimalLabel = svmtestLabel
            pcaPreDataResult = pcaPreData
            print 'i=%s>>> \t' % i,lineLen,int(minSumErr),1-minErr
        
        
        target_names = [str(i) for i in list(set(y_test))]
        print '包含的种类：',target_names
        print classification_report(y_test,optimalLabel,target_names = target_names)
        print '特征数量=%s , 存在最优解：>> \t' % optimalNum,lineLen,int(minSumErr),1-minErr
        return optimalSVMClf, pcaPreDataResult
        
            
            

# 存储模型
def storeModel(model, filename):
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


# 加载模型
def getModel(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def trainDRSVM():
    startTime = time.time()

    # 加载数据
    trainData, trainLabel, preData = opencsv()
    # 模型训练 (数据预处理-降维)
    optimalSVMClf, pcaPreData = getOptimalAccuracy(trainData, trainLabel, preData)

    storeModel(optimalSVMClf, os.path.join(data_dir, 'output/Result_sklearn_SVM.model'))
    storeModel(pcaPreData, os.path.join(data_dir, 'output/Result_sklearn_SVM.pcaPreData'))

    print("finish!")
    stopTime = time.time()
    print('TrainModel store time used:%f s' % (stopTime - startTime))


def preDRSVM():
    startTime = time.time()
    # 加载模型和数据
    optimalSVMClf = getModel(os.path.join(data_dir, 'output/Result_sklearn_SVM.model'))
    pcaPreData = getModel(os.path.join(data_dir, 'output/Result_sklearn_SVM.pcaPreData'))

    # 结果预测
    testLabel = optimalSVMClf.predict(pcaPreData)
    # print("testLabel = %f" % testscore)
    # 结果的输出
    saveResult(testLabel, os.path.join(data_dir, 'output/Result_sklearn_SVM.csv'))
    print("finish!")
    stopTime = time.time()
    print('PreModel load time used:%f s' % (stopTime - startTime))
    return pcaPreData,testLabel


# 数据可视化
def dataVisulization(data, labels):
    pca = PCA(n_components=2, whiten=True) # 使用PCA方法降到2维
    pca.fit(data)
    pcaData = pca.transform(data)
    uniqueClasses = set(labels)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cClass in uniqueClasses:
        plt.scatter(pcaData[labels==cClass, 0], pcaData[labels==cClass, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('MNIST visualization')
    plt.show()


if __name__ == '__main__':
    trainData, trainLabel, preData = opencsv()
#     dataVisulization(trainData, trainLabel)


    # 训练并保存模型
    trainDRSVM()

    # 分析数据
    analyse_data(trainData)
    # 加载预测数据集
    
    pcaPreData,testLabel = preDRSVM()
    dataVisulization(pcaPreData,testLabel)
    
