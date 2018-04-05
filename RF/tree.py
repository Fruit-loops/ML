# coding: UTF-8
# cart decision tree

from math import pow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
def make_feat():
    '''
    数据来自uci数据集，车分类的一个数据集

        | class values

        unacc, acc, good, vgood

        | attributes

        buying:   vhigh, high, med, low.
        maint:    vhigh, high, med, low.
        doors:    2, 3, 4, 5more.
        persons:  2, 4, more.
        lug_boot: small, med, big.
        safety:   low, med, high.

    :return: data(feat+label)

    '''

    buying = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
    maint = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
    doors = {'2': 0, '3': 1,'4': 2, '5more': 3}
    persons = {'2':0,'4':1,'more':2}
    lug_boot = {'small':0,'med':1,'big':2}
    safety = {'low':0,'med':1,'high':2}
    label = {'unacc':1,'acc':2,'good':3,'vgood':4}
    feature =[buying,maint,doors,persons,lug_boot,safety,label]
    file = open('car.data')
    data = []
    for line in file.readlines():
        lst = line.strip('\n').split(',')
        sample_temp = []
        for i in range(len(lst)):
            sample_temp.append(feature[i][lst[i]])
        data.append(sample_temp)
    data = np.array(data)
    return data



class node:
    def __init__(self,fea=-1, value=None, results=None,left=None,right=None):
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left
def label_uniq_cnt(data):
    '''
    统计数据集中不同的类标签label的个数
    :param data: 原始数据集
    :return: 样本中标签的个数
    '''
    #print(type(data))
    label_uniq_cnt = {}
    for x in data:
        #print(data)
 
        label = x[-1]
        #print(label)
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] += 1
    return label_uniq_cnt
def cal_gini_index(data):
    '''

    :param data(list):数据集
    :return: gini(float)
    '''
    total_sample = len(data)
    if len(data) == 0:
        return 0
    label_counts = label_uniq_cnt(data)
    # 计算数据集的gini指数
    gini = 0
    for label in label_counts:
        gini += pow(label_counts[label],2)
    gini = 1- float(gini)/pow(total_sample,2)
    return gini

'''
停止划分的条件：1结点中的样本数小于给定的阈值。2样本集的gini指数小于给定阈值（基本属于同一类样本）3没有更多特征
'''

def split_tree(data,fea,value):
    '''

    :param data: 训练集（包含了label)
    :param fea: 需要划分的特征index
    :param value: 要指定划分的值
    :return: (set1,set2) tuple 划分的两个子集
    '''
    set_1 = []
    set_2 = []
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return (set_1,set_2)


def build_tree(data):
    '''

    :param data(list):train data
    :return: node
    '''
    if len(data) == 0:
        return node()
    # 1 计算当前gini指数
    currentGini = cal_gini_index(data)
    bestGain = 0.0
    bestCriteria = None  #储存最佳切分属性和最佳切分点
    bestSets = None #储存切分后的两个数据集

    feature_num = len(data[0]) - 1 #样本中的特征个数
    # 2 找到最好的划分
    for fea in range(0,feature_num):
        feature_values = {}
        #取得fea特征处的每一个取值
        for sample in data:
            feature_values[sample[fea]] = 1
        
        #针对每一个取值，尝试将数据集划分，并计算gini指数
        for value in feature_values.keys():
            #根据fea特征中的value将数据集划分成左右子树
            (set_1,set_2) = split_tree(data,fea,value)
            #计算当前gini指数
            nowGini = float(len(set_1)*cal_gini_index(set_1) + len(set_2)*cal_gini_index(set_2)) / len(data)
            #计算gini指数的增加量
            gain = currentGini - nowGini
            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea,value)
                bestSets = (set_1,set_2)
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0],value=bestCriteria[1],right=right,left=left)
    else:
        results = label_uniq_cnt(data)

        return node(results = sorted(results.items(),key = lambda x:x[1],reverse = True)[0][0]) #返回label中数目最多的一类

def predict_for_one(sample,tree):
    '''
    对每个样本进行预测
    :param sample:  一个样本
    :param tree: 构建好的树
    :return: 预测结果
    '''
    
    if tree.results != None:
        return tree.results
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict_for_one(sample,branch)

def predict(data,tree):
    '''
    对数据集进行预测
    param: dataL mat m*n
    param: tree 构建好的决策树
    return: ndarray
    '''
    m = len(data)
    pred = np.zeros((m,1))
    for i in range(m):
        pred[i,0] = predict_for_one(data[i],tree)
    return pred            

if __name__ == '__main__':
    data = make_feat()
    #print(data)
    data_train,data_test = train_test_split(data,shuffle=True,train_size=0.8,test_size=0.2,random_state=555)
    tree = build_tree(data_train)
    test_pred = predict(data_test,tree)
    accuracy = accuracy_score(data_test[:,-1],test_pred)
    print(accuracy)
    
    



