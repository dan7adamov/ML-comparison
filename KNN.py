#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gzip
from sklearn.model_selection import train_test_split as tts

uspsTrain = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.train\\zip.train', delimiter = " ", autostrip = True)
uspsTest = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.test\\zip.test', delimiter = " ", autostrip = True)
usps = np.concatenate((uspsTrain, uspsTest))

X_trainUsps, X_testUsps, y_trainUsps, y_testUsps = tts(usps[:,1:], usps[:, 0], test_size=0.125, random_state=709)


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


def neighArray(X_testSample, X_train, y_train, neighNumb):
    nearDist = {} # dictionary with key as X_train index and value as Eucledian distance
    for i in range(X_train.shape[0]):
        eucSum = np.linalg.norm(X_train[i,:] - X_testSample)
        if len([*nearDist]) < neighNumb: # when nearDist is empty we fill it up with first elements just to make it full
            nearDist[i] = eucSum
        else:
            maxDistKeyVal = max(nearDist, key=nearDist.get), max(nearDist.values())
            if eucSum < maxDistKeyVal[1]:
                del nearDist[maxDistKeyVal[0]]
                nearDist[i] = eucSum
    return [*nearDist]

def knnClassifier(X_testSample, X_train, y_train, neighNumb):
    nearDist = neighArray(X_testSample, X_train, y_train, neighNumb)
    knnLabelVote = {}
    for indX in nearDist:
        label = y_train[indX]
        if label in [*knnLabelVote]:
            knnLabelVote[label] += 1
        else:
            knnLabelVote[label] = 1
    return max(knnLabelVote, key = knnLabelVote.get)

def knnRegressor(X_testSample, X_train, y_train, neighNumb):
    nearDist = neighArray(X_testSample, X_train, y_train, neighNumb)
    knnLabelSum = 0
    for indX in nearDist:
        knnLabelSum += y_train[indX]
    return knnLabelSum / len(nearDist)

def errorRate(X_testPredicted, y_test):
    totalErrors = 0
    for t in zip(X_testPredicted, y_test):
        if t[0] != t[1]:
            totalErrors += 1
    return totalErrors/len(X_testPredicted)

def errorRateWrapper(X_train, X_test, y_train, y_test, neighNumb):
    predictions = []
    for testSample in X_test:
        predictions.append(knnClassifier(testSample, X_train, y_train, neighNumb))
    e = errorRate(predictions, y_test)
    return e

def errorRateDifNeighWrapper(X_train, X_test, y_train, y_test, neighNumbRange):
    for r in range(1, neighNumbRange + 1):
        print("For", r , "nearest neighbours in KNN the error rate is:", errorRateWrapper(X_train, X_test, y_train, y_test, r))


# In[2]:


errorRateDifNeighWrapper(X_trainUsps, X_testUsps, y_trainUsps, y_testUsps, 10)


# In[6]:


errorRateDifNeighWrapper(X_trainMnist, X_testMnist[:20], y_trainMnist, y_testMnist[:20], 10)


# In[7]:


from PIL import Image
import matplotlib.pyplot as plt

for k in range(0, len(X_testUsps)):
    knnC = knnClassifier(X_testUsps[k], X_trainUsps, y_trainUsps, 1)
    if knnC != y_testUsps[k]:
        plt.imshow((X_testUsps[k,:]*127.5+128).astype(int).reshape(16,16))
        print("prediction:", knnC)
        print("label:", y_testUsps[k], "n:", k)
        break


# In[20]:


for k in range(19, len(X_testMnist)):
    knnC = knnClassifier(X_testMnist[k], X_trainMnist, y_trainMnist, 1)
    if knnC != y_testMnist[k]:
        plt.imshow((X_testMnist[k,:]).astype(int).reshape(28,28))
        print("prediction:", knnC)
        print("label:", y_testMnist[k], "n:", k)
        break


# In[ ]:




