#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
from sklearn.model_selection import train_test_split as tts

uspsTrain = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.train\\zip.train', delimiter = " ", autostrip = True)
uspsTest = np.genfromtxt('C:\\Users\\Dan Adamov\\Desktop\\RHUL\\3rd Year\\CS3920\\zip.test\\zip.test', delimiter = " ", autostrip = True)
usps = np.concatenate((uspsTrain, uspsTest))

X_trainUsps, X_testUsps, y_trainUsps, y_testUsps = tts(usps[:,1:], usps[:, 0], test_size=0.125, random_state=709)


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

