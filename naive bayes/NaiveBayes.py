#coding=utf-8
#naive Baeys
#2018.01.10
'''
使用贝叶斯过滤文本的一个小例子
'''
from numpy import *
import codecs

#词表到向量的转换函数
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    return returnVec

#训练算法，从词向量计算概率
def trainNB0(trainMatrix,trainCategory):
    '''

    :param tarinMatrix: 已经转换的文档矩阵
    :param trainCategory: 分类向量
    :return:p0Vect:分类为0的特征概率向量P(X=x^1,x^2...x^len(trainMatrix)|Y=0)
            p1Vect: 分类为1
            pAbusive: p(y=1)

    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory/float(numTrainDocs))
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrixp[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

'''
p(w0|1)p(w1|1)p(w2|1) 其中如果有0，则会影响结果
所以可以将所有词的出现次数初始化为1，分母初始化为2
另一个遇到的问题是下溢出，可以用取对数的方法来避免
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''

    :param vec2Classify: 要预测的向量
    :param p0Vec: train得到的三个向量
    :param p1Vec: train得到的三个向量
    :param pClass1: train得到的三个向量
    :return:
    '''
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
#定义一个词袋模型
def bagOfWords2VecMN(vocabList,inputset):
    returnVec = [0]*len(vocabList)
    for word in vocabList:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#定义一个切分文本的函数
def testParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = [] ;classList = []; fullText = []
    for i in range(1,26):
        wordList = testParse(open(r'email\spam\\'+str(i)+'.txt').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = testParse(open(r'email\ham\\'+str(i)+'.txt').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorcount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorcount += 1
    print('the error rate is: ',float(errorcount)/len(testSet))


if __name__ == '__main__':
    spamTest()

