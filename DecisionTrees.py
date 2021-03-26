#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import operator as op
import gzip
from sklearn.model_selection import train_test_split as tts

uspsTrain = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.train\\zip.train', delimiter = " ", autostrip = True)
uspsTest = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.test\\zip.test', delimiter = " ", autostrip = True)
usps = np.concatenate((uspsTrain, uspsTest))
X_trainUsps, X_testUsps, y_trainUsps, y_testUsps = tts(usps[:,1:], usps[:, 0], test_size=0.125, random_state=709)
# Dowloaded and parsed usps dataset into train and test datasets(ratio 8:1 respectively) and have separate arrays for features and their corresponding labels

X_trainMnist = np.array([])
y_trainMnist = np.array([])
X_testMnist = np.array([])
y_testMnist = np.array([])
with gzip.open('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\Individual Project\\train-images-idx3-ubyte.gz', 'rb') as trainSampleFile:
    trainSampleBuffer = trainSampleFile.read()
    X_trainMnistUnshaped = np.frombuffer(trainSampleBuffer, dtype = np.uint8, offset = 16)
    X_trainMnist = X_trainMnistUnshaped.reshape(60000, 784)
with gzip.open('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\Individual Project\\train-labels-idx1-ubyte.gz', 'rb') as trainLabelFile:
    trainLabelBuffer = trainLabelFile.read()
    y_trainMnist = np.frombuffer(trainLabelBuffer, dtype = np.uint8, offset = 8)
with gzip.open('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\Individual Project\\t10k-images-idx3-ubyte.gz', 'rb') as testSampleFile:
    testSampleBuffer = testSampleFile.read()
    X_testMnistUnshaped = np.frombuffer(testSampleBuffer, dtype = np.uint8, offset = 16)
    X_testMnist = X_testMnistUnshaped.reshape(10000, 784)
with gzip.open('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\Individual Project\\t10k-labels-idx1-ubyte.gz', 'rb') as testLabelFile:
    testLabelBuffer = testLabelFile.read()
    y_testMnist = np.frombuffer(testLabelBuffer, dtype = np.uint8, offset = 8)
# Dowloaded Mnist dataset into train and test datasets(ratio 6:1 respectively) and have separate arrays for features and their corresponding labels

treeLabels = 0 , 1
maxTreeDepth = 3

class TreeLabelWrapper:
    def __init__(self, labels):
        self.labels = labels
        treeLabels = labels
        self.tree = DecisionTree()

class DecisionTree:
    def __init__(self, featureNmbr = None, featureThreshold = None, predictedLabel = None):
        self.featureNmbr = featureNmbr
        self.featureThreshold = featureThreshold
        self.predLabel = predictedLabel
        self.right = None
        self.left = None
        
    def treeFactory(self, X_train, y_train, path):
        if len(path) >= maxTreeDepth:
            return
        featNum, featThr, labelIndices, errorRate = self.featureSelector(X_train, y_train, path)
        self.featureNmbr, self.featureThreshold = featNum, featThr
        self.right = DecisionTree(None, None, treeLabels[labelIndices[0]])
        self.left = DecisionTree(None, None, treeLabels[labelIndices[1]])
        self.right.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__ge__)])
        self.left.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__lt__)])
    
    def classifier(sample):
        curNode = self
        if curNode.right and curNode.left:
            if sample[self.featureOfSample]>= self.featureThreshold:
                curNode = self.right
            if sample[self.featureOfSample]< self.featureThreshold:
                curNode = self.left
        else:
            if sample[self.featureOfSample]>= self.featureThreshold:
                return #correect label
            if sample[self.featureOfSample]< self.featureThreshold:
                return #correct label

    def featureThresholdSelectorV2(self, X_train, y_train, featureNmbr, path):

        instancesOfFeatureLabel = np.zeros(256)
        instancesOfFeatureNotLabel = np.zeros(256)
        cumulativeThreshold = np.zeros(256)

        for i in range(X_train.shape[0]):
            goodSample = True
            if path:
                for t in path: # t is a tuple -> (featureNumber, featureThreshold, operator)
                    if not t[2] (X_train[i, t[0]], t[1]):
                        goodSample = False
                        break
                if not goodSample:
                    continue
                
            # Processing only good samples
            if y_train[i] == treeLabels[0]:
                instancesOfFeatureLabel[X_train[i,featureNmbr]] += 1
            elif y_train[i] == treeLabels[1]:
                instancesOfFeatureNotLabel[X_train[i,featureNmbr]] += 1

        cumSumFeature = np.cumsum(instancesOfFeatureLabel[::-1])[::-1]
        cumSumNotFeature = np.cumsum(instancesOfFeatureNotLabel[::-1])[::-1]

        np.subtract(cumSumFeature, cumSumNotFeature, cumulativeThreshold)
        featureThreshold = np.argmax(abs(cumulativeThreshold))

        if cumulativeThreshold[featureThreshold] >= 0:
            indicesOfTreeLabel = 0, 1
            errorRate = 1 - cumSumFeature[featureThreshold] / cumSumFeature[0]
        else:
            indicesOfTreeLabel = 1, 0
            errorRate = 1 - cumSumNotFeature[featureThreshold] / cumSumNotFeature[0]

        return featureThreshold, indicesOfTreeLabel, errorRate
    
    
    def featureSelector(self, X_train, y_train, path):
        leastErrorRateFeatureIndex = 0
        leastErrorRate = 1.0000000001
        leastErrorRateFeatureThreshold = None
        curFeatureLabelIndices = None
        
        for featNbr in range(X_train.shape[1]):
        # for featNbr in range(120, 140): # DEBUG, HARDCODE
            if path and featNbr != path[-1][0] or not path:
                curFeature = self.featureThresholdSelectorV2(X_train, y_train, featNbr, path)
                if curFeature[2] < leastErrorRate:
                        leastErrorRateFeatureIndex = featNbr
                        leastErrorRateFeatureThreshold = curFeature[0]
                        curFeatureLabelIndices = curFeature[1]
                        leastErrorRate = curFeature[2]
        
        return leastErrorRateFeatureIndex, leastErrorRateFeatureThreshold, curFeatureLabelIndices, leastErrorRate


# In[24]:


test = DecisionTree()
test.treeFactory(X_trainMnist, y_trainMnist, [])


# curFeatureLabelIndices returned by feature selector has an error arate of 100% and no new value is assigned

# In[27]:


test = DecisionTree()


# In[28]:


test.featureThresholdSelectorV2(X_trainMnist, y_trainMnist, 208, None)


# In[29]:


testInstOfFeatureLab = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
testCumSumFeature = np.cumsum(testInstOfFeatureLab[::-1])[::-1]
print(testCumSumFeature)


# In[30]:


test.featureSelector(X_trainMnist, y_trainMnist, None)


# So the error is with the error rates. When featThreshold of 0 is selelcted. That makes both arrays select all samples at zero and it somehow has a zero percent error rate
