from python_speech_features import mfcc
import scipy.io as wavfile as wavfile
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random
import math
import operator

def getNeighbors(trainingSet,instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = math.dist(trainingSet[x], instance,k) + math.dist(instance, trainingSet[x],k)
        distances.append((trainingSet[x][2], dist))
        distances.sort()
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1
        else:
            classVote[response]=1
    sorter = sorted(classVote.items(), key = operator)
    return sorter [0][0]

def getaccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct +=1
    return 1*correct/len(testSet)

    directory = "__path_to_dataset__"
    open("my_dat", 'wb') as f
    i = 0

    for folder in os.listdir(directory):
        i +=1
        if i ==11:
            break
        for file in os.listdir(directory+folder):
            (rate,sig) = wav.read(directory+folder+"/"+file)
            mfcc_feat = mfcc(sig,rate, winlen = 0.02, appenEnergy = False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
    f.close()
    dataset = []
def loadDataset(filename, split, trSet, teSet):
    with open("my.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
            f.close()
            break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.sppend(dataset[x])
        teSet.append(dataset[x])

trainingSet = []
testSet = []
loadDataset("my.dat", 0.66, trainingSet, testSet)

leng = len(testSet)
predictions.append(nearestClass(getNeighbors(trainingSet,testSet[x],5)))

accuracy = getAccuracy(testSet, predictions)
print(accuracy)
