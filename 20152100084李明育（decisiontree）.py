from numpy import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import copy
import json

np.seterr(divide='ignore', invalid='ignore')

class DTree(object):
    def __init__(self):  # 构造方法
        self.tree = {}  # 生成树
        self.dataSet = []  # 数据集
        self.labels = []  # 标签集


    # 数据导入函数
    def loadDataSet(self, path, labels):
        recordList = []
        fp = open(path, "r")  # 读取文件内容
        content = fp.read()
        fp.close()
        rowList = content.splitlines()  # 按行转换为一维表
        recordList = [row.split("\t") for row in rowList if row.strip()]  # strip()函数删除空格、Tab等
        self.dataSet = recordList
        self.labels = labels


    # 执行决策树函数
    def train(self):
        labels = copy.deepcopy(self.labels)
        self.tree = self.buildTree(self.dataSet, labels)


    # 构件决策树：穿件决策树主程序
    def buildTree(self, dataSet, lables):
        cateList = [data[-1] for data in dataSet]  # 抽取源数据集中的决策标签列
        # 程序终止条件1：如果classList只有一种决策标签，停止划分，返回这个决策标签
        if cateList.count(cateList[0]) == len(cateList):
            return cateList[0]
        # 程序终止条件2：如果数据集的第一个决策标签只有一个，返回这个标签
        if len(dataSet[0]) == 1:
            return self.maxCate(cateList)
        # 核心部分
        bestFeat, featValueList= self.getBestFeat(dataSet)  # 返回数据集的最优特征轴
        bestFeatLabel = lables[bestFeat]
        tree = {bestFeatLabel: {}}
        del (lables[bestFeat])
        for value in featValueList:  # 决策树递归生长
            subLables = lables[:]  # 将删除后的特征类别集建立子类别集
            # 按最优特征列和值分隔数据集
            splitDataset = self.splitDataSet(dataSet, bestFeat, value)
            subTree = self.buildTree(splitDataset, subLables)  # 构建子树
            tree[bestFeatLabel][value] = subTree
        return tree


    # 计算出现次数最多的类别标签
    def maxCate(self, cateList):
        items = dict([(cateList.count(i), i) for i in cateList])
        return items[max(items.keys())]


    # 计算最优特征
    def getBestFeat(self, dataSet):
        Num_Feats = len(dataSet[0][:-1])
        totality = len(dataSet)
        BaseEntropy = self.computeEntropy(dataSet)
        ConditionEntropy = []     # 初始化条件熵
        slpitInfo = []    # for C4.5,caculate gain ratio
        allFeatVList = []
        for f in range(Num_Feats):
            featList = [example[f] for example in dataSet]
            [splitI, featureValueList] = self.computeSplitInfo(featList)
            allFeatVList.append(featureValueList)
            slpitInfo.append(splitI)
            resultGain = 0.0
            for value in featureValueList:
                subSet = self.splitDataSet(dataSet, f, value)
                appearNum = float(len(subSet))
                subEntropy = self.computeEntropy(subSet)
                resultGain += (appearNum/totality)*subEntropy
            ConditionEntropy.append(resultGain)    # 总条件熵
        infoGainArray = BaseEntropy*ones(Num_Feats)-array(ConditionEntropy)
        infoGainRatio = infoGainArray/array(slpitInfo)  # C4.5信息增益的计算
        bestFeatureIndex = argsort(-infoGainRatio)[0]
        return bestFeatureIndex, allFeatVList[bestFeatureIndex]

    # 计算划分信息
    def computeSplitInfo(self, featureVList):
        numEntries = len(featureVList)
        featureVauleSetList = list(set(featureVList))
        valueCounts = [featureVList.count(featVec) for featVec in featureVauleSetList]
        pList = [float(item)/numEntries for item in valueCounts]
        lList = [item*math.log(item, 2) for item in pList]
        splitInfo = -sum(lList)
        return splitInfo, featureVauleSetList


 # 计算信息熵
    # @staticmethod
    def computeEntropy(self, dataSet):
        dataLen = float(len(dataSet))
        cateList = [data[-1] for data in dataSet]  # 从数据集中得到类别标签
        # 得到类别为key、 出现次数value的字典
        items = dict([(i, cateList.count(i)) for i in cateList])
        infoEntropy = 0.0
        for key in items:  # 香农熵： = -p*log2(p) --infoEntropy = -prob * log(prob, 2)
            prob = float(items[key]) / dataLen
            infoEntropy -= prob * math.log(prob, 2)
        return infoEntropy


    # 划分数据集： 分割数据集； 删除特征轴所在的数据列，返回剩余的数据集
    # dataSet : 数据集； axis: 特征轴； value: 特征轴的取值
    def splitDataSet(self, dataSet, axis, value):
        rtnList = []
        for featVec in dataSet:
            if featVec[axis] == value:
                rFeatVec = featVec[:axis]  # list操作：提取0~（axis-1）的元素
                rFeatVec.extend(featVec[axis + 1:])   # 将特征轴之后的元素加回
                rtnList.append(rFeatVec)
        return rtnList


def main():     #生成决策树模型
    dtree = DTree()
    dtree.loadDataSet("fruit2.txt",['mass', 'width', 'height', 'color_score'])
    dtree.train()
    print (dtree.tree)   
    print('\n')  
    #将决策树结构写入fruits.txt文件中  
    with open('fruits.txt',"w") as fp:
        fp.write(json.dumps(dtree.tree))

def run():    #求模型准确率
    data = pd.read_table('fruit.txt', error_bad_lines=False)
    ch = ['mass', 'width', 'height', 'color_score']
    x = data[ch]
    y = data['fruit_label']
    
    #创建训练集和测试集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
    
    #将缩放比例扩展应用到我们为训练集计算的测试集上
    from sklearn.preprocessing  import MinMaxScaler
    sc = MinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    #调用函数求准确率
    from sklearn.tree import DecisionTreeClassifier
    trees = DecisionTreeClassifier().fit(x_train, y_train)
    print('决策树学习算法的测试集的准确率为: {:.3f}'
     .format(trees.score(x_test, y_test)))
    

    
if __name__=='__main__':
    main()
    run()
    
