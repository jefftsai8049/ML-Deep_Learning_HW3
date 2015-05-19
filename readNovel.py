__author__ = 'jefftsai'

import csv

class readNovel:
    # def __init__(self):
    #     # open pre-processing raw file




    def findStrat(self):
        # print("find start")
        # find where is the article start
        i=0
        while 1:
            str = self.file.readline()
            # find does the line exist "Chapter of CHAPTER"
            if str.find("Chapter") >= 0 or str.find("CHAPTER") >= 0:
                break
            else:
                self.useless.write(str+"\n")
            i = i+1
            # if read 300 lines,but dosn't find the start of the article
            if i>300:
                break
        # print(i)
        return i+1

    def findLines(self,inFileName):
        file = open(inFileName,"r")
        i=0
        while 1:
            text = file.readline()
            if len(text) < 1:
                break
            i = i+1
        return i
    def readSplit(self ,inFileName, outFileName, useless):
        self.file = open(inFileName, "r", errors='ignore')
        # for save ignore text
        self.useless = open(useless,"w")
        # for save remain text
        outFile = open(outFileName,"w")
        # find start position
        self.findStrat()

        # initialize
        string = ""
        oldText = ""
        j=0
        while 1:
            # read one line
            text = self.file.readline()

            # replace exception condition
            content = text.replace("\n"," ")
            content = content.replace("Mr.","Mr")
            content = content.replace("Mrs.","Mrs")
            content = content.replace("--"," ")
            content = content.replace("-"," ")
            content = content.replace('(',' ')
            content = content.replace(')',' ')
            content = content.replace("---"," ")
            content = content.replace(","," ")

            # if EOF break out the while loop
            if text == "":
                break

            # if this line is the start of the article
            if content.find("Chapter") >= 0 or content.find("CHAPTER") >= 0 :
                content = " "
            # if this line is the end of the article
            if content.find("End of") >= 0 or content.find("THE END") >= 0 or content.find("*END") >= 0:
                self.findStrat()

            # if two lines is \n, change line
            if oldText == "\n" and text == "\n":
                string = self.checkString(string)
                string = string.lower()
                outFile.write(string+"\n")
                string = ""

            # read the text char by char
            else:
                for i in range(0,len(content)):
                    if content[i] !="." and content[i] !="?" and content[i] !="!":
                        if (ord(content[i]) > 64 and ord(content[i]) < 91) or (ord(content[i]) > 96 and ord(content[i]) < 123) or content[i] == " ":
                            string = string+content[i]
                    else:

                        string = self.checkString(string)
                        string = string.lower()
                        outFile.write(string+"\n")
                        string = ""
            oldText = text
            print(j)
            j = j+1
            # if j > 1000:
            #     break

        outFile.close()
        self.useless.close()
        self.file.close()
        return True

    def checkString(self,string):


        # string = string.replace('\"','')
        # # string = string.replace("\'",'')
        # string = string.replace(',',' ,')
        # string = string.replace(';',' ;')
        # string = string.replace('(',' ')
        # string = string.replace(')',' ')
        # string = string.replace("---"," ")
        # string = string.replace("--"," ")
        # string = string.replace("-"," ")
        if len(string) > 0:
            if string[0] == " " or string[0] == "\'":
                if len(string) == 1:
                    string = ""
                else:
                    string = string[1:]
                    if string[0] == " " or string[0] == "\'":
                        string = string[1:]
            # print(string)
            # print(len(string))


        # print(string)
        return string

    def preprocessing(self,filename,outFileName):
        splitData = open(filename, "r")
        outFile = open(outFileName, "w")
        # j=0
        while 1:
            string = splitData.readline()
            if len(string) > 1:
                string = string.replace("\n", "")
                if string[0] == "":
                    while 1:
                        string = string[1:]
                        if string[0] !=" ":
                            break
                else:
                    string = string.lower()
                    data = string.split(" ")
                    for i in range(0,data.count("")):
                        data.remove("")
                    # print(data)
                    # outFile.write(str(j)+",")
                    outFile.write("START,")
                    outFile.write(",".join(data))
                    outFile.write(",END"+"\n")
                    # j =j +1

            elif len(string) == 0:
                break
        return True

    def word2index(self,inFileName,outFileName,mapFileName):
        inFile = open(inFileName,"r")
        outFile = open(outFileName,"w")

        map = dict([["START",0],["END",1]])


        i = 2
        k = 0
        while 1:
            string = inFile.readline()
            if len(string) < 1:
                break
            string = string.replace("\n","")
            data = string.split(",")
            # print(data)
            for j in range(0,len(data)):
                if ~(data[j] in map):
                    if map.setdefault(data[j],i) == i:
                        i = i+1
            k = k+1
            # if k > 3:
            #     break
        print(map)
        self.writeDict(mapFileName,map)

        inFile.seek(0)
        n=0
        while 1:
            string = inFile.readline()
            string = string.replace("\n","")
            data = string.split(",")

            temp = ""
            if len(string) < 1:
                break
            else:
                # data = map[data]
                # print(data)
                for m in range(0,len(data)):
                    data[m] = map[data[m]]
                    temp = temp + str(data[m])+","
                temp = temp[:-1]+"\n"
                outFile.write(temp)
            n = n +1
            # if n > 3:
            #     break
        # print(data)
        outFile.close()
        return True

    def writeDict(self,mapFileName,map):
        mapFile = open(mapFileName,"w")
        for key, val in map.items():
            mapFile.write(key+","+str(val)+"\n")
        return True

    def loadDict(self,mapFileName):

        mapFile = open(mapFileName,"r")
        map = dict()
        while 1:
            text = mapFile.readline()
            if len(text) < 1:
                break
            else:
                mapElement = text.split(",")
                map.setdefault(mapElement[0],int(mapElement[1]))
        return map

    def word2VectorIndex(self,mapFileName,trainOutFileName,trainIndexFileName):
        map = self.loadWord2VectorMap(mapFileName)

        inFile = open(trainOutFileName,"r")
        outFile = open(trainIndexFileName,"w")

        j=0
        while 1:
            # if j > 5:
            #     break
            # j = j +1

            text = inFile.readline()
            string = ""
            if len(text)<2:
                break
            else:
                text = text[:-1]
                data = text.split(",")

                for i in range(0,len(data)):
                    if i == 0:
                        data[i] = -2
                    elif i == len(data)-1:
                        data[i] = -3
                    else:
                        if data[i] in map:
                            data[i] = map[data[i]]
                        else:
                            data[i] = -1
                    string = string+str(data[i])+","
                string = string[:-1]
                outFile.write(string+"\n")




        inFile.close()
        outFile.close()

        return "word2vector index ok"

    def loadWord2VectorMap(self,mapFileName):
        mapFile = open(mapFileName,"r")
        set = []

        i = 0
        while 1:
            text = mapFile.readline()
            if len(text)<1:
                break
            else:
                text = text[:-1]
                element = text.split(" ")
                element[1] = int(element[1])
                set.append(element)

            # if i > 10:
            #     break
            # i=i+1
        # print(set)
        map = dict(set)
        return map