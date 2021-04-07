#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import operator as op
import gzip
from sklearn.model_selection import train_test_split as tts

# uspsTrain = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.train\\zip.train', delimiter = " ", autostrip = True)
# uspsTest = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.test\\zip.test', delimiter = " ", autostrip = True)
# usps = np.concatenate((uspsTrain, uspsTest))
# X_trainUsps, X_testUsps, y_trainUsps, y_testUsps = tts(usps[:,1:], usps[:, 0], test_size=0.125, random_state=709)
# Dowloaded and parsed usps dataset into train and test datasets(ratio 8:1 respectively) and have separate arrays for features and their corresponding labels

X_trainMnist = np.array([])
y_trainMnist = np.array([])
X_testMnist = np.array([])
y_testMnist = np.array([])
with gzip.open('C:\\Users\\danad\\Personal Project\\Individual Project\\train-images-idx3-ubyte.gz', 'rb') as trainSampleFile:
    trainSampleBuffer = trainSampleFile.read()
    X_trainMnistUnshaped = np.frombuffer(trainSampleBuffer, dtype = np.uint8, offset = 16)
    X_trainMnist = X_trainMnistUnshaped.reshape(60000, 784)
with gzip.open('C:\\Users\\danad\\Personal Project\\Individual Project\\train-labels-idx1-ubyte.gz', 'rb') as trainLabelFile:
    trainLabelBuffer = trainLabelFile.read()
    y_trainMnist = np.frombuffer(trainLabelBuffer, dtype = np.uint8, offset = 8)
with gzip.open('C:\\Users\\danad\\Personal Project\\Individual Project\\t10k-images-idx3-ubyte.gz', 'rb') as testSampleFile:
    testSampleBuffer = testSampleFile.read()
    X_testMnistUnshaped = np.frombuffer(testSampleBuffer, dtype = np.uint8, offset = 16)
    X_testMnist = X_testMnistUnshaped.reshape(10000, 784)
with gzip.open('C:\\Users\\danad\\Personal Project\\Individual Project\\t10k-labels-idx1-ubyte.gz', 'rb') as testLabelFile:
    testLabelBuffer = testLabelFile.read()
    y_testMnist = np.frombuffer(testLabelBuffer, dtype = np.uint8, offset = 8)
# Dowloaded Mnist dataset into train and test datasets(ratio 6:1 respectively) and have separate arrays for features and their corresponding labels

treeLabels = None # FIX THIS TO NONE AFTER BUG TESTING
maxTreeDepth = 5
allTrees = []

def treesGenerator():
    for i in range(10):
        for j in range(i + 1, 10):
            allTrees.append(TreeLabelWrapper((i, j)))
            
def classifyAllTrees(X_train, y_train):
    for t in allTrees:
        t.treeFactory(X_train, y_train, [])

# wrapper class facilitates generation of decision trees with different label tuples
class TreeLabelWrapper:
    def __init__(self, labels):
        self.labels = labels
        self.tree = DecisionTree()
        
    def treeFactory(self, X_train, y_train):
        global treeLabels
        treeLabels = self.labels
        self.tree.treeFactory(X_train, y_train, [])
        

# main algorithm logic class. Creates instances of tree nodes, and hondles all the splitting
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
        # do not create 'empty' nodes with uninformative divisions, or inadequate number of samples in the subset
        
#         if featThr == -1: # no samples for right node
#             self.left = DecisionTree(None, None, treeLabels[labelIndices[0]]) # treeLabels need to be handled by treeIndices
#             return
#         if featThr == -2: # no samples for left node
#             self.right = DecisionTree(None, None, treeLabels[labelIndices[0]])
#             return
        if featThr < 0:            
            return
        if featNum is None:
            return
        
        self.featureNmbr, self.featureThreshold = featNum, featThr
        self.right = DecisionTree(None, None, treeLabels[labelIndices[0]])
        self.left = DecisionTree(None, None, treeLabels[labelIndices[1]])
        self.right.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__ge__)])
        self.left.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__lt__)])
    
    def classifier(self, sample):
        
        if self.featureThreshold and sample[self.featureNmbr] >= self.featureThreshold:
            if self.right:
                return self.right.classifier(sample)
            else:
                return self.predLabel
            
        if self.featureThreshold and sample[self.featureNmbr] < self.featureThreshold:
            if self.left:
                return self.left.classifier(sample)
            else:
                return self.predLabel
        else:
            return self.predLabel

        
    def classifierV2(self, sample):
        if self.featureThreshold is None:
             return predLabel
        else:
            if self.featureThreshold and sample[self.featureNmbr] >= self.featureThreshold:
                return self.right.classifier(sample)
            if self.featureThreshold and sample[self.featureNmbr] < self.featureThreshold:
                return self.left.classifier(sample)
        
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
        
        if cumSumFeature[0] == 0 and cumSumNotFeature[0] == 0: # Extra test, should never happen
            print("Possible Bug - no samples in subset")
            return 0, (0, 1), 1.1
        if cumSumFeature[0] == 0: # nothing to split, only one type of labels
            return -1, (1, 0), 1.1
        if cumSumNotFeature[0] == 0: # nothing to split, only one type of labels
            return -2, (0, 1), 1.1
        
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
        for featNbr in range(X_train.shape[1]):
        # for featNbr in (300, 305, 310, 315, 320, 325, 330, 335, 340): # DEBUG, HARDCODE
            if path and featNbr != path[-1][0] or not path:
                curFeature = self.featureThresholdSelectorV2(X_train, y_train, featNbr, path)
                if curFeature[0] <= 0 and leastErrorRate == 1:
                    leastErrorRateFeatureIndex = None
                    leastErrorRateFeatureThreshold = curFeature[0]
                    curFeatureLabelIndices = curFeature[1]
                    # No perform return, it leaves the loop if condition is satisfied.
                if curFeature[2] < leastErrorRate:
                        leastErrorRateFeatureIndex = featNbr
                        leastErrorRateFeatureThreshold = curFeature[0]
                        curFeatureLabelIndices = curFeature[1]
                        leastErrorRate = curFeature[2]
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
        print()
        print("depth =", depth)
        if self.subSetVolume:
            print("volume =", self.subSetVolume)
        print(self.featureNmbr, self.featureThreshold, self.predLabel)
        for subTree in (self.left, self.right):
            if subTree:
                subTree.auditFull(depth + 1)


# In[12]:


testWrapper = TreeLabelWrapper( (7, 2) )
testWrapper.treeFactory(X_trainMnist[100:200], y_trainMnist[100:200])


# In[13]:


testWrapper.tree.auditFull()


# In[19]:


y_trainMnist[200:220]


# In[17]:


testWrapper.tree.classifierV2(X_trainMnist[117])


# In[22]:


from PIL import Image
import matplotlib.pyplot as plt

# k = 117
for k in range(100, 250):
    if y_trainMnist[k] in [2, 7]:
        pred = testWrapper.tree.classifierV2(X_trainMnist[k])
        if pred != y_trainMnist[k]:
            # plt.imshow((X_trainMnist[k,:]).astype(int).reshape(28,28))
            print("prediction:", pred)
            print("label:", y_trainMnist[k])


# In[ ]:




