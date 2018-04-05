#coding:UTF-8

import svm as Svm
import numpy as np

def load_data_libsvm(data_file):
    '''
    导入训练数据
    :param data_file:
    :return: data(mat)
             label(mat)
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        lines = line.strip().split(' ')
        #提取label
        label.append(float(lines[0]))
        #提取特征并放到矩阵中
        index = 0
        tmp = []
        for i in range(1,len(lines)):
            li = lines[i].strip().split(':')
            if int(li[0]) -1 == index:
                tmp.append(float(li[1]))
            else:
                while(int(li[0]) - 1 > index):
                    tmp.append(0)
                    index += 1
                tmp.append(float(li[1]))
            index += 1
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)
    f.close()
    return np.mat(data),np.mat(label).T




if __name__ == '__main__':
    #1.导入训练数据
    print('-------------------- 1. load data -------------------')
    dataSet,labels = load_data_libsvm('heart_scale')

    #2.训练svm模型
    print('-------------------- 2. training  -------------------')

    C = 0.65
    toler = 0.001
    maxIter = 1000
    svm_model = Svm.SVM_training(dataSet,labels,C,toler,maxIter)
    #计算准确性
    print('-------------------- 3. cal accuracy ----------------')
    accuracy = Svm.cal_accuracy(svm_model, dataSet, labels)
    print('the training accuracy is : %.3f%%' % (accuracy * 100))
    #保存
    print('-------------------- 4. save ------------------------')
    Svm.save_svm_model(svm_model,'model_file')





