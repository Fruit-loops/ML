#coding=utf-8
#linear regression classification
#2018.01.11
from numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradDscent(dataMatIn, classLabels,alpha=0.001,maxCycles=5000):
    '''

    :param dataMatIn: 特征矩阵
    :param classLabels: label 向量
    :param alpha: 步长 学习率
    :param maxCycles: 最大循环数
    :param weights: 初始权重
    :return: W
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # m*n * n*1 = m * 1
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

### plot
def plotBestFit(para):
    import matplotlib.pyplot as plt
    parameter = para
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-parameter[0]-parameter[1]*x)/parameter[2]
    ax.plot(x,y)
    plt.show()
# sgd
def stocGradDscent(dataMatrix,classLabels,numIter=1):
    dataMatrix = array(dataMatrix)
    classLabels = array(classLabels)
    m,n = shape(dataMatrix)
    weights = ones(n)

    for j in range(numIter):
        dataIndex = [k for k in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[i]*weights)) # 一次一个样本 1*n * n*1 数值
            error = classLabels[i] - h
            #print(type(dataMatrix))
            weights = weights + alpha * error * dataMatrix[i]  # 1*1 * 1*1 * 1*n
            del(dataIndex[randIndex])
    return weights

## a practice
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))

    return 1.0 if prob>0.5 else 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradDscent(trainingSet,trainingLabels)

    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():

        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    #print(numTestVec)
    errorRate = (float(errorCount)/numTestVec)
    print ('the error rate of this test is : %f' % errorRate)
    frTrain.close()
    frTrain.close()
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average errror rate is: %f" %(numTests,errorSum/float(numTests)))


if __name__ == '__main__':
    multiTest()





