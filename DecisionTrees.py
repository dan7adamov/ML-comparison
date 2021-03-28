#!/usr/bin/env python
# coding: utf-8

# In[30]:


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
maxTreeDepth = 5

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
        self.subSetVolume = None
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
    
    def classifier(self, sample):
        
        if self.featureThreshold != 0 and sample[self.featureNmbr] >= self.featureThreshold:
            if self.right:
                return self.right.classifier(sample)
            else:
                return self.predLabel
        else:
            return self.predLabel
            
        if self.featureThreshold and sample[self.featureNmbr] < self.featureThreshold:
            if self.left:
                return self.left.classifier(sample)
            else:
                return self.predLabel
        else:
            return self.predLabel

    def featureThresholdSelectorV2(self, X_train, y_train, featureNmbr, path):
        
        indicesOfTreeLabel = 0, 1
        errorRate = 1
        
        self.subSetVolume = 0
        
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
                self.subSetVolume += 1
            elif y_train[i] == treeLabels[1]:
                instancesOfFeatureNotLabel[X_train[i,featureNmbr]] += 1
                self.subSetVolume += 1

        cumSumFeature = np.cumsum(instancesOfFeatureLabel[::-1])[::-1]
        cumSumNotFeature = np.cumsum(instancesOfFeatureNotLabel[::-1])[::-1]
        
        if cumSumFeature[0] == 0 and cumSumNotFeature[0] == 0:
            return 0, (0, 1), 1.1

        np.subtract(cumSumFeature, cumSumNotFeature, cumulativeThreshold)
        featureThreshold = np.argmax(abs(cumulativeThreshold))
        
        if featureThreshold != 0:
            if cumulativeThreshold[featureThreshold] >= 0:
                indicesOfTreeLabel = 0, 1
                errorRate = 1 - cumSumFeature[featureThreshold] / cumSumFeature[0]
            else:
                indicesOfTreeLabel = 1, 0
                errorRate = 1 - cumSumNotFeature[featureThreshold] / cumSumNotFeature[0]

        return featureThreshold, indicesOfTreeLabel, errorRate
    
    
    def featureSelector(self, X_train, y_train, path):
        leastErrorRateFeatureIndex = 0
        leastErrorRate = 1
        leastErrorRateFeatureThreshold = 0
        curFeatureLabelIndices = 0, 1
        
        # for featNbr in range(X_train.shape[1]):
        for featNbr in (300, 320, 340, 360, 380, 420, 440, 460): # DEBUG, HARDCODE
            if path and featNbr != path[-1][0] or not path:
                curFeature = self.featureThresholdSelectorV2(X_train, y_train, featNbr, path)
                if curFeature[2] < leastErrorRate:
                        leastErrorRateFeatureIndex = featNbr
                        leastErrorRateFeatureThreshold = curFeature[0]
                        curFeatureLabelIndices = curFeature[1]
                        leastErrorRate = curFeature[2]
        if curFeatureLabelIndices is None:
            print("Fix this bug")
        return leastErrorRateFeatureIndex, leastErrorRateFeatureThreshold, curFeatureLabelIndices, leastErrorRate

    
    def audit(self, sample):
        print(self.featureNmbr, self.featureThreshold, self.predLabel)
        if self.featureThreshold and sample[self.featureNmbr] >= self.featureThreshold:
            if self.right:
                return self.right.audit(sample)
        if self.featureThreshold and sample[self.featureNmbr] < self.featureThreshold:
            if self.left:
                return self.left.audit(sample)
                   
    
    def auditFull(self, depth = 0):
        print("depth =", depth)
        if self.subSetVolume:
            print("volume =", self.subSetVolume)
        print(self.featureNmbr, self.featureThreshold, self.predLabel)
        for subTree in (self.left, self.right):
            if subTree:
                subTree.auditFull(depth + 1)
        


# In[31]:


test = DecisionTree()
test.treeFactory(X_trainMnist, y_trainMnist, [])


# In[32]:


test.auditFull()


# In[5]:


test.featureThresholdSelectorV2(X_trainMnist, y_trainMnist, 208, None)


# In[48]:


testInstOfFeatureLab2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 19, 47, 20, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testInstOfFeatureLab1 = [0, 0, 0, 0, 0, 0, 14, 19, 47, 20, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testCumSumFeature1 = np.cumsum(testInstOfFeatureLab1[::-1])[::-1]
print(testCumSumFeature1)
testCumSumFeature2 = np.cumsum(testInstOfFeatureLab2[::-1])[::-1]
np.subtract(testCumSumFeature1, testCumSumFeature2, testCumSumFeature)
# featureThreshold = np.argmax(abs(cumulativeThreshold))
print(testCumSumFeature)
print(np.argmax(abs(testCumSumFeature)))


# In[55]:


test.featureSelector(X_trainMnist, y_trainMnist, None)


# In[12]:


test.left.predLabe
test.right.predLabel


# So the error is with the error rates. When featThreshold of 0 is selelcted. That makes both arrays select all samples at 0 threshold thus it has 0.00 error rate. Making method work incorect as no differentiation is made between two sets of different samples.
