__author__ = 'jefftsai'

import readNovel

novel = readNovel.readNovel()

print("Read and Split Data : " + str(novel.readSplit("train_data/training.txt","train_data/training_split.txt", "train_data/useless.txt")))
print("Preprocessing Data :　" + str(novel.preprocessing("train_data/training_split.txt", "train_data/training_out.csv")))

print(novel.word2index("train_data/training_out.csv","train_data/training_index.csv","train_data/train_map.csv"))

# print(novel.loadWord2VectorMap("train_data/word2vec/training_classes_500.txt"))