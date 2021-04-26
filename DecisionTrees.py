import numpy as np
import operator as op
import gzip
import pickle as pkl
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

# global variables
treeLabels = None
maxTreeDepth = 10

# generates all possible tuples with different label combinations and makes an empty tree with it
def treesGenerator():
    global allTrees
    allTrees = []
    for i in range(10):
        for j in range(i + 1, 10):
            allTrees.append(TreeLabelWrapper((i, j)))

# starts classifing samples and making divisions to form a complete DecisionTree
def classifyAllTrees(X_train, y_train):
    for t in allTrees:
        t.treeFactory(X_train, y_train)

def classifyFromTree(X_train, y_train, treeIndex, treeFinalIndex):
    for i in range(treeIndex, treeFinalIndex+1):
        allTrees[i].treeFactory(X_train, y_train)
        pkl.dump( allTrees, open( "allTrees-{0:0{1}}".format(i, 3), "wb" ) )

# returns the predicted label for a given sample, by looking at which label is most common from all possible Trees combinations
def decisionMaker(sample):
    predLabels = []
    for t in allTrees:
        predLabels.append( t.tree.classifierV2(sample) )
    return max(set(predLabels), key = predLabels.count)

# method for finding the error rate
def errorRate(xTestSet, yTestSet):
        numOfErrors = 0
        for k in range(xTestSet.shape[0]): # testSet should be an np.array of rank 2
            predLabel = decisionMaker(xTestSet[k])
            if predLabel != yTestSet[k]:
                numOfErrors += 1
        return numOfErrors / xTestSet.shape[0]
    

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
        self.right = None
        self.left = None
        self.subSetVolume = None # debug variable - number of samples in the current node
        
    # method for creating instances of DecisionTree which are nodes in our tree, and splitting the dataset into child nodes
    def treeFactory(self, X_train, y_train, path):
        if len(path) >= maxTreeDepth:
            return
        featNum, featThr, labelIndices, errorRate = self.featureSelector(X_train, y_train, path)
        
        # do not create 'empty' nodes with uninformative divisions, or inadequate number of samples in the subset
        if featThr < 0 or featNum is None:            
            return
        
        # divison is informative so two new nodes are created and their parent node contains the split information
        self.featureNmbr, self.featureThreshold = featNum, featThr
        self.right = DecisionTree(None, None, treeLabels[labelIndices[0]])
        self.left = DecisionTree(None, None, treeLabels[labelIndices[1]])
        self.right.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__ge__)])
        self.left.treeFactory(X_train, y_train, path + [(featNum, featThr, op.__lt__)])
        
    
    # method for selecting the most informative threshold of a given feature number
    def featureThresholdSelectorV3(self, X_train, y_train, featureNmbr, path):
        
        indicesOfTreeLabel = 0, 1
        errorRate = 1
        
        # debug variable - number of samples in the current node
        self.subSetVolume = 0
        
        instancesOfFeatureLabel = np.zeros(256)
        instancesOfFeatureNotLabel = np.zeros(256)
        
        # throwing away samples which dont fit earlier constraints of the nodes
        for i in range(X_train.shape[0]):
            goodSample = True
            if path:
                for t in path: # t is a tuple -> (featureNumber, featureThreshold, operator)
                    if not t[2] (X_train[i, t[0]], t[1]):
                        goodSample = False
                        break
                if not goodSample:
                    continue
                    
            # Processing only samples with 2 correct labels
            if y_train[i] == treeLabels[0]:
                instancesOfFeatureLabel[X_train[i,featureNmbr]] += 1
                self.subSetVolume += 1
            elif y_train[i] == treeLabels[1]:
                instancesOfFeatureNotLabel[X_train[i,featureNmbr]] += 1
                self.subSetVolume += 1
        
        # ammount of samples of each label type in the current node
        sumLabel = np.sum(instancesOfFeatureLabel)
        sumNotLabel = np.sum(instancesOfFeatureNotLabel)
               
        # uninformative splits are prevented
        if sumLabel == 0 and sumNotLabel == 0: # test for when max depth is reached
            # print("Possible Bug - no samples in subset or max depth is reached by the tree")
            return 0, (0, 1), 1.1
        if sumLabel == 0: # nothing to split, only one type of labels
            return -1, (1, 0), 1.1
        if sumNotLabel == 0: # nothing to split, only one type of labels
            return -2, (0, 1), 1.1
        
        # find the most common greyscale number
        featureArgmax = np.argmax(instancesOfFeatureLabel)
        notFeatureArgmax = np.argmax(instancesOfFeatureNotLabel)
        
        # find the mean of the two numbers, which is a threshold
        featureThreshold = round((featureArgmax + notFeatureArgmax) / 2)
        
        # cumilitive array is calculated for finding error rate
        cumSumFeature = np.cumsum(instancesOfFeatureLabel[::-1])[::-1]
        cumSumNotFeature = np.cumsum(instancesOfFeatureNotLabel[::-1])[::-1]
        
        # the order of labels is decided and the error rate is given
        if featureThreshold != 0:
            if featureArgmax >= notFeatureArgmax:
                indicesOfTreeLabel = 0, 1
                errorRate = 1 - cumSumFeature[featureThreshold] / cumSumFeature[0]
            else:
                indicesOfTreeLabel = 1, 0
                errorRate = 1 - cumSumNotFeature[featureThreshold] / cumSumNotFeature[0]

        return featureThreshold, indicesOfTreeLabel, errorRate
    
    
    # method for selecting a feature number with most informative division
    def featureSelector(self, X_train, y_train, path):
        leastErrorRateFeatureIndex = 0
        leastErrorRateFeatureThreshold = 0
        curFeatureLabelIndices = 0, 1
        leastErrorRate = 1
        
        # iterates through all features
        for featNbr in range(X_train.shape[1]):
        # for featNbr in (300, 305, 310, 315, 320, 325, 330, 335, 340): # DEBUG
        
            # prevents algorithm from slecting the same feature number for two consecutive nodes
            if path and featNbr != path[-1][0] or not path:
                # featureThreshold, indicesOfTreeLabel, errorRate = curFeature
                curFeature = self.featureThresholdSelectorV3(X_train, y_train, featNbr, path)
                
                # catches split which is not informative (featureThreshold = 0)
                if curFeature[0] == 0:
                    continue
                
                # node with a subset containing only one label
                if curFeature[0] < 0:
                    return None, curFeature[0], curFeature[1], 0
                
                # good splits are slowly improved, by selecting the feature number with least error rate
                if curFeature[2] < leastErrorRate:
                    leastErrorRateFeatureIndex = featNbr
                    leastErrorRateFeatureThreshold = curFeature[0]
                    curFeatureLabelIndices = curFeature[1]
                    leastErrorRate = curFeature[2]
                   
        return leastErrorRateFeatureIndex, leastErrorRateFeatureThreshold, curFeatureLabelIndices, leastErrorRate
    
    
    # method for getting a predicted label from a given sample from the Tree we built
    def classifierV2(self, sample):
        
        # if that is a node with 100% accuracy (only samples of one label are left in this node)
        if self.featureThreshold is None:
            return self.predLabel
        
        # this is satisfied if this is the last node in the tree, but because of depth constraint, we cannot go any deeper
        elif self.right is None and self.left is None:
            return self.predLabel
        
        # recursion
        else:
            if sample[self.featureNmbr] >= self.featureThreshold:
                return self.right.classifierV2(sample)
            if sample[self.featureNmbr] < self.featureThreshold:
                return self.left.classifierV2(sample)
    
    # debug function - prints all the tree information from its nodes
    def auditFull(self, depth = 0):
        print()
        print("depth =", depth)
        if self.subSetVolume:
            print("volume =", self.subSetVolume)
        print(self.featureNmbr, self.featureThreshold, self.predLabel)
        for subTree in (self.left, self.right):
            if subTree:
                subTree.auditFull(depth + 1)