__author__ = 'jefftsai'

import numpy as np
import math
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
        self.momentum = 1


        print("Initializing Neural Network Parameters ...\n")
        NNParameters = RNNParameters.RNNParameters(inputLayerSize, hiddenLayerSize, memorySize, labelNum)
        trainData = self.loadTrainData(trainDataFileName)

        print("Feedforward Using Neural Network ...\n")
        wholeSentence = self.N_Matrix(trainData,word2vecSize)
        # self.wordsNum = len(trainData)
        X = wholeSentence[:,:-1]
        Y = wholeSentence[:,1:]

        XR = np.fliplr(X)
        YR = np.fliplr(Y)

        currentWord = 1
        j,gradinet = self.nnCostFunction(NNParameters, inputLayerSize, hiddenLayerSize, memorySize, labelNum, X, Y,self.momentum,currentWord)
        print(X)
        print(Y)


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

    def ReLU(self,z):
        for i in range(0,len(z)):
            if z[i] < 0:
                z[i] = 0

        return z

    def softMax(self,z):

        expZ = np.exp(z)
        expZSum = np.sum(z)
        return expZ/expZSum
        # exp_z = exp(z);     % exponential values of each z
        # exp_z_sum = sum(exp_z);     % sum of exp_z along comlumn
        # y = exp_z / exp_z_sum;      % y should be values between 0~1

    def nnCostFunction(self, parameters, inputLayerSize, hiddenLayerSize, memorySize, lableNum, X, Y, momentum, currentWord):
        currentIndex = 1
        currentSize = hiddenLayerSize*inputLayerSize
        wordsNum = X.shape[1]+1
        J = 0

        grad = RNNParameters.RNNParameters

        aM = np.zeros((memorySize,1),dtype=np.float64)
        memoryAll = np.zeros((memorySize,wordsNum-1),dtype=np.float64)
        zAll = np.zeros((memorySize,wordsNum-1),dtype=np.float64)

        for i in range(0,wordsNum-1):
            XCurrentWord = X[:,i:i+1]
            YCurrentWord = Y[:,i:i+1]
            z1 = np.dot(parameters.Wm,aM)+np.dot(parameters.W1,XCurrentWord)+parameters.bm+parameters.b1
            zAll[:,i:i+1] = z1
            a1 = self.ReLU(z1)
            aM = a1
            memoryAll[:,i:i+1] = aM
            zOut = np.dot(parameters.Wo,a1)+parameters.bo
            aOut = self.softMax(zOut)

            if i == wordsNum-i:
                YOut = aOut

            tHat = (YCurrentWord==1)
            Yt = aOut[tHat]

            if Yt == 0:
                currentWordCost = -math.log(0.0001)
            else:
                currentWordCost = -math.log(Yt)

            J = J+currentWordCost

        startMemory = wordsNum-currentWord-1-3
        if startMemory == 0:
            z1BPTT = np.zeros((memorySize,1))
            layer1 = np.zeros((memorySize,1))
        else:
            z1BPTT = zAll[:,startMemory:startMemory+1]
            layer1 = memoryAll[:,startMemory:startMemory+1]







        return grad,J
