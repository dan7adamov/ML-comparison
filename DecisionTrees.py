#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

treeLabels = None

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
        
    def addRight(self, featNumb, featThresh):
        self.right = DecisionTree(self.labels, path + [(self.featNumb, self.featThresh, labels)])
    
    def addLeft(self, featNumb, featThresh):
        self.left = DecisionTree(self.labels, path + [(self.featNumb, self.featThresh, (labels[1], labels[0]) )])
    
    def add(self, curFeatureNumber, curFeatureThreshold, curLabels):
        if curFeatureThreshold >= self.featureThreshold:
            if self.right is None:
                self.right = Tree(curFeatureNmbr,curFeatureThreshold, curLabels)
                return
            self.right.add(curFeatureNmbr,curFeatureThreshold, curLabels)
            return
        if curFeatureThreshold < featureThreshold:
            if self.left is None:
                self.left = Tree(curFeatureNmbr,curFeatureThreshold, curLabels)
                return
            self.left.add(curFeatureNmbr,curFeatureThreshold, curLabels)
            return
        
    
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

    def featureThresholdSelectorV2(X_train, y_train, featureNmbr, path):

        instancesOfFeatureLabel = np.zeros(256)
        instancesOfFeatureNotLabel = np.zeros(256)
        cumulativeThreshold = np.zeros(256)

        for i in range(X_train.shape[0]):
            goodSample = True
            for t in path: # t is a tuple -> (featureNumber, featureThreshold, operator)
                if ! t[2] (X_train[i, t[0]], t[1]):
                    goodSample = False
                    break
            if !goodSample:
                continue
                
            # Processing only good samples
            if y_train[i] == labels[0]:
                instancesOfFeatureLabel[X_train[i,featureNmbr]] += 1
            elif y_train[i] == labels[1]:
                instancesOfFeatureNotLabel[X_train[i,featureNmbr]] += 1

            cumSumFeature = np.cumsum(instancesOfFeatureLabel[::-1])[::-1]
            cumSumNotFeature = np.cumsum(instancesOfFeatureNotLabel[::-1])[::-1]

        np.subtract(cumSumFeature, cumSumNotFeature, cumulativeThreshold)
        featureThreshold = np.argmax(abs(cumulativeThreshold))

        if cumulativeThreshold[featureThreshold] >= 0:
            resultLabels = labels
            errorRate = 1 - cumSumFeature[featureThreshold] / cumSumFeature[0]
        else:
            resultLabels = labels[1], labels[0]
            errorRate = 1 - cumSumNotFeature[featureThreshold] / cumSumNotFeature[0]

        return featureThreshold, resultLabels, errorRate
    
    
    def featureSelector(X_train, y_train, path):
        leastErrorRateFeatureIndex = 0
        leastErrorRate = 1.0
        curFeatureLabels = None
        # for featNbr in range(X_train.shape[1]):
        for featNbr in range(120, 140): # DEBUG, HARDCODE
            if self.path and featNbr != self.path[-1][0] or !self.path:
                curFeature = featureThresholdSelectorV2(X_train, y_train, featNbr, path)
                if curFeature[2] < leastErrorRate:
                        leastErrorRateFeatureIndex = featNbr
                        leastErrorRate = curFeature[2]
                        curFeatureLabels = curFeature[1]
        return leastErrorRateFeatureIndex, curFeatureLabels, leastErrorRate

def featureThresholdSelector(X_train, y_train, desiredLabel, comparisonLabel, featureNmbr):
    instancesOfFeatureLabel = np.zeros(256) # change 256 so it selects the range of the features in the set
    instancesOfFeatureNotLabel = np.zeros(256)
    cumulativeThreshold = np.zeros(256)
    for i in range(X_train.shape[0]):
        if y_train[i] == desiredLabel:
            instancesOfFeatureLabel[X_train[i,featureNmbr]] += 1
        elif y_train[i] == comparisonLabel:
            instancesOfFeatureNotLabel[X_train[i,featureNmbr]] += 1
        cumSumFeature = np.cumsum(instancesOfFeatureLabel[::-1])[::-1]
        cumSumNotFeature = np.cumsum(instancesOfFeatureNotLabel[::-1])[::-1]
    np.subtract(cumSumFeature, cumSumNotFeature, cumulativeThreshold)
    featureThreshold = np.argmax(abs(cumulativeThreshold))
    if cumulativeThreshold[featureThreshold] >= 0:
        labels = desiredLabel, comparisonLabel
        errorRate = 1 - cumSumFeature[featureThreshold] / cumSumFeature[0]
    else:
        labels = comparisonLabel, desiredLabel
        errorRate = 1 - cumSumNotFeature[featureThreshold] / cumSumNotFeature[0]
    # print(cumulativeThreshold)
    return featureThreshold, labels , errorRate


# In[22]:


featureThresholdSelector(X_trainMnist, y_trainMnist, y_trainMnist[1], y_trainMnist[5], 128)


# In[24]:


print(y_trainMnist[1], y_trainMnist[5])
featureOfSampleSelector(X_trainMnist, y_trainMnist, 136, y_trainMnist[1], y_trainMnist[5])


# In[5]:


X_trainMnist.shape


# In[12]:


print(np.amax(X_trainMnist[1]))
print(np.amin(X_trainMnist[1]))
print()
print(np.amax(X_trainUsps[1]))
print(np.amin(X_trainUsps[1]))

# PERHAPS TURN NP.ZEROS TO A DICTIONARY SO I DO NOT HAVE TO DO WEIRD NORMALISATIONS 


# In[ ]:




