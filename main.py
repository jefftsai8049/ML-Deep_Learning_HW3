__author__ = 'jefftsai'

import readNovel

novel = readNovel.readNovel()
# for read raw data and remove useless word
# print("Read and Split Data : " + str(novel.readSplit("train_data/training.txt","train_data/training_split.txt", "train_data/useless.txt")))

# convert to lower and remove empty line
# print("Preprocessing Data :　" + str(novel.preprocessing("train_data/training_split.txt", "train_data/training_out.csv")))

# convert to index (current no use)
# print(novel.word2index("train_data/training_out.csv","train_data/training_index.csv","train_data/train_map.csv"))

# convert to index(by google code word2vec)
print(novel.word2VectorIndex("train_data/word2vec/word2vec_map_100.txt","train_data/training_out.csv","train_data/training_index_word2vec100.csv"))
