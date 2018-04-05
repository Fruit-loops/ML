# coding:utf=8
# 2018.01.16
# Adaboost using a level decision tree

from numpy import *
# one level decision tree
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    '''

    :param dataMatrix(mat):  训练数据
    :param dimen(int): 要划分的特征
    :param threshVal: 选择的划分值
    :param threshIneq: '< or >'
    :return: retArray(list) 返回一个分类结果
    '''
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    '''
    构建树桩
    :param dataArr(mat) 训练数据
    :param classLabels(list) 标签
    :param D 数据的权重向量
    :return: bestStump(dict) 包含了选择树桩的属性（选择的特征，划分值，ineq）
             minError
             bestClassEst 最好的树桩分类
    '''
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = float('inf')
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal= (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 # array
                weightedError = D.T * errArr
                print(weightedError)
                print('split dim %d, thresh %.2f, thresh ineqal: %s, the weighed error is %.3f' %\
                      (i,threshVal,inequal,weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClassEst


def adaBoostTrain(dataArr,classLabels,numIt=1):
    '''
    训练算法
    :param dataArr(mat): 训练数据
    :param classLabels(list): 标签
    :param numIt(int):迭代次数
    :return: 应该返回一个 alpha 向量 和 矩阵
    '''
    dataArr = mat(dataArr)
    #classLabels = mat(classLabels)
    weakClassArr = []
    m = shape(dataArr)[0]

    D = mat(ones((m,1))/m)
    #print(D)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print('D:',D.T)
        alpha = float(0.5*log((1.0-error)/max(error,float(1e-16)))) # 防止下溢
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst:',mat(classLabels).T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        #print('aggClassEst:',aggClassEst.T)

        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))

        errorRate = aggErrors.sum()/m
        print('total error:',errorRate,'\n')
        if errorRate == 0.0:break
    return weakClassArr

#using adaboost to predict

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():

        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

if __name__ == '__main__':
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoostTrain(datArr,labelArr,10)


