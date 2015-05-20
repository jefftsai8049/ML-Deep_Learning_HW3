__author__ = 'jefftsai'
import numpy as np

class RNNParameters:
    def __init__(self, inputLayerSize, hiddenLayerSize, memorySize, labelNum,):

        self.W1 = self.randInitializeWeights(hiddenLayerSize,inputLayerSize)
        self.b1 = self.randInitializeWeights(hiddenLayerSize,1)
        self.Wm = self.randInitializeWeights(memorySize,hiddenLayerSize)
        self.bm = self.randInitializeWeights(memorySize,1)
        self.Wo = self.randInitializeWeights(labelNum,hiddenLayerSize)
        self.bo = self.randInitializeWeights(labelNum,1)

    def randInitializeWeights(self,inLayer,outLayer):

        out = np.random.random([inLayer,outLayer])/100

        return out