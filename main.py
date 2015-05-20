__author__ = 'jefftsai'

import readNovel
import RNN as ML


# Pre-processing
novel = readNovel.readNovel()

# for read raw data and remove useless word
# print("Read and Split Data : " + str(novel.readSplit("train_data/training.txt","train_data/training_split.txt", "train_data/useless.txt")))

# convert to lower and remove empty line
# print("Preprocessing Data :　" + str(novel.preprocessing("train_data/training_split.txt", "train_data/training_out.csv")))

# convert to index (current no use)
# print(novel.word2index("train_data/training_out.csv","train_data/training_index.csv","train_data/train_map.csv"))

# convert to index(by google code word2vec)
# print(novel.word2VectorIndex("train_data/word2vec/word2vec_map_100_filter.txt","train_data/training_out.csv","train_data/training_index_word2vec100_filter.csv"))

# print(novel.findLines("train_data/train_filter_word2vec100.csv"))

# 500 corresponds to -3 ~ 499, so input layer should have 503 nodes
word2vecSize = 100+3
# 503 nodes
vectorSize = word2vecSize
# 1006 hidden units
inputLayerSize = word2vecSize
# 1006 hidden units
hiddenLayerSize = (word2vecSize) * 2
# 1006 hidden units
memorySize = hiddenLayerSize
# 503 labels
numLabels = word2vecSize
# 503 labels
labelNum = word2vecSize

trainDataFileName = "train_data/training_index_word2vec100.csv"
sequenceNum = 2365657

model = ML.RNN(trainDataFileName, sequenceNum, word2vecSize, inputLayerSize, hiddenLayerSize, memorySize, labelNum)
