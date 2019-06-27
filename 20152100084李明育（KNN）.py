import csv
import random
import math
import operator

#加载进来的特征数据从字符串转换成整数。然后我们随机地切分训练数据集和测试数据集
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    #使用open方法打开文件，并用csv模块读取数据
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
  
#只需要包含测量好的4个维度，这4个维度放在数组的前几个位置。限制维度的一个办法就是增加一个参数，告诉函数前几个维度需要处理，忽略后面的维度。   
def euclideanDistance(instance1, instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#计算待预测数据到所有数据实例的距离，取其中距离最小的N个
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#让这些近邻元素来对预测属性投票，得票最多的选项作为预测结果。
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

#计算在测试数据集中算法正确预测的比例，这个比例叫分类准确度
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    if float(len(testSet)):
        return (correct/float(len(testSet))) * 100.0
    else:
        return 0

def main():
    trainingSet = []
    testSet = []
    split = 0.70
    loadDataset('fruit.csv',split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set:' + repr(len(testSet)))
    predictions=[]
    k = 5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) +',actual=' +repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:' + repr(accuracy) + '%')
        
if __name__=='__main__':
    main()