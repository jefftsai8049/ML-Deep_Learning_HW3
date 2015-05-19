__author__ = 'jefftsai'
import numpy as np

class RNNParameters:
    def __init__(self, inputLayerSize, hiddenLayerSize, memorySize, labelNum,):

        self.W1,self.b1 = self.randInitializeWeights(inputLayerSize,hiddenLayerSize)
        self.Wm,self.bm = self.randInitializeWeights(memorySize,hiddenLayerSize)
        self.Wo,self.bo = self.randInitializeWeights(hiddenLayerSize,labelNum)

    def randInitializeWeights(self,inLayer,outLayer):

        W = np.random.random([inLayer,outLayer])/100
        b = np.random.random([inLayer,1])/100

        return W,b