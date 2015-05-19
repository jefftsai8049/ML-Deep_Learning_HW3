__author__ = 'jefftsai'

import numpy as np
# import scipy.sparse as sparse

import RNNParameters




class RNN:
    def __init__(self, trainDataFileName, sentenceNum, word2vecSize, inputLayerSize, hiddenLayerSize, memorySize, labelNum):
        print("Prameters Setting ...\n")
        self.vectorSize = word2vecSize
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.memorySize = memorySize
        self.sentenceNum = sentenceNum
        self.labelNum = labelNum

        self.trainDataFileName = trainDataFileName

        self.learningRate = 1
        self.yita = 0.001
        self.decay = 0.9999


        print("Initializing Neural Network Parameters ...\n")
        Parameters = RNNParameters.RNNParameters(inputLayerSize, hiddenLayerSize, memorySize, labelNum)
        trainData = self.loadTrainData(trainDataFileName)

        print("Feedforward Using Neural Network ...\n")
        wholeSentence = self.N_Matrix(trainData,word2vecSize)


    def setParameters(self,learningRate = 1, yita = 0.001, decay = 0.9999):
        self.learningRate = learningRate
        self.yita = yita
        self.decay = decay


    def loadTrainData(self,trainDataFileName):
        trainDataFile = open(trainDataFileName, "r")
        trainDataStr = trainDataFile.readline()
        trainDataStr = trainDataStr[:-1]
        trainDataList = trainDataStr.split(",")

        trainData = np.array(trainDataList,dtype=np.float64)

        return trainData


    def N_Matrix(self,trainData,word2vecSize):
        wordsNum = len(trainData)
        wordsIndex = np.add(trainData,3,dtype=np.int)
        cols = np.linspace(0,wordsNum-1,wordsNum,dtype=np.int)
        wholeSentence = np.zeros((word2vecSize,wordsNum),dtype=np.int)
        wholeSentence[wordsIndex,cols] = 1

        return wholeSentence
