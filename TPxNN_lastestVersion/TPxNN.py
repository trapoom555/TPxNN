import random
from math import exp
import numpy as np 
from math import floor

#Matrix Manipulation
class Matrix:
    def __init__(self,x,y):
        self.matrix = [[random.uniform(-1,1) for i in range(y)] for j in range(x)]

def multiply(m1,m2):
        if(len(m1[0]) == len(m2)):
            c = Matrix(len(m1),len(m2[0]))
            for i in range(len(m1)):
                for k in range(len(m2[0])):
                    sum = 0
                    for j in range(len(m1[0])):
                        sum += m1[i][j]*m2[j][k]
                    c.matrix[i][k] = sum
            return(c.matrix)            
        else:
            print("Error : Can't multiply Matrices")

def sMultiply(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] * m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't sMultiply Matrices")

def scalarMultiply(n,m):
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = m[i][j] * n
    return(m)

def add(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] + m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't add Matricies")

def subtract(m1,m2):
    if(len(m1) == len(m2) and len(m1[0]) == len(m2[0])):
        c = Matrix(len(m1),len(m1[0]))
        for i in range(len(m1)):
            for j in range(len(m1[0])):
                c.matrix[i][j] = m1[i][j] - m2[i][j]
        return(c.matrix)
    else:
        print("Error : Can't subtract Matricies")

def transpose(m):
    c = Matrix(len(m[0]),len(m))
    for j in range(len(m)):
        for i in range(len(m[0])):
            c.matrix[i][j] = m[j][i]
    return(c.matrix)

#NN Function
def activationFunction(m1):
    c = Matrix(len(m1),len(m1[0]))
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            c.matrix[i][j] = 1/(1+exp(-m1[i][j]))
    return(c.matrix)


def dSigmoid(m1):
    c = Matrix(len(m1),len(m1[0]))
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            c.matrix[i][j] = m1[i][j] * ( 1 - m1[i][j])
    return(c.matrix)

#InputData manipulation
def declareTrainingAndTestingDataObj(arr):
    for i in range(len(arr)):
        arr[i] = TrainingAndTestingData()

def setTargetAndLabel(TrainingAndTestingData,target,label):
    c = []
    for i in range(len(TrainingAndTestingData)):
        c.append(0)
    declareTrainingAndTestingDataObj(c)
    for i in range(len(TrainingAndTestingData)):
        c[i].input = TrainingAndTestingData[i]
        c[i].target = target
        c[i].label = label
    return(c)

#Merge,Shuffle,Normalizing,Grouping
def MSNGData(GroupOfDataInList,divider,percentofTrainingData):
    c = []
    inputdata = []
    targetdata = []
    labeldata = []
    for i in range(len(GroupOfDataInList)):
        c = np.concatenate((c,GroupOfDataInList[i]))
    np.random.shuffle(c)
    for i in range(len(c)):
        for j in range(len(c[i].input)):
            c[i].input[j] /= divider
    numberOfTrainingData = floor(len(c) * percentofTrainingData / 100)
    for i in range(len(c)):
        inputdata.append(c[i].input)
        targetdata.append(c[i].target)
        labeldata.append(c[i].label)
    TrainingInput = inputdata[:numberOfTrainingData]
    TrainingTarget = targetdata[:numberOfTrainingData]
    TrainingLabel = labeldata[:numberOfTrainingData]
    TestingInput = inputdata[numberOfTrainingData:]
    TestingTarget = targetdata[numberOfTrainingData:]
    TestingLabel = labeldata[numberOfTrainingData:]
    return(TrainingInput,TrainingTarget,TrainingLabel,TestingInput,TestingTarget,TestingLabel)


class TrainingAndTestingData:
    def __init__(self):
        self.input = []
        self.target = []
        self.label = 0
#Core
class NeuralNetwork:
    def __init__(self,listofarchitecture):
        self.lr = 0.1
        self.achitecture = listofarchitecture
        self.inputTraining = []
        self.targetTraining = []
        self.w = [0 for i in range(len(listofarchitecture) - 1)]
        self.data = [0 for i in range(len(listofarchitecture))]
        self.datab = [0 for i in range(len(listofarchitecture))]
        self.bias = [0 for i in range(len(listofarchitecture) - 1)]
        self.error = [0 for i in range(len(listofarchitecture) - 1)]
        self.deltaw = [0 for i in range(len(listofarchitecture) - 1)]
        self.deltab = [0 for i in range(len(listofarchitecture) - 1)]
        for layer in range(len(self.achitecture)):
            self.data[layer] = Matrix(self.achitecture[layer],1)
            self.datab[layer] = Matrix(self.achitecture[layer],1)
        for layer in range(len(self.achitecture) - 1):
            self.w[layer] = Matrix(self.achitecture[layer+1],self.achitecture[layer])
            self.bias[layer] = Matrix(self.achitecture[layer+1],1)
        for layer in range(len(self.achitecture) - 1):
            self.error[layer] = Matrix(self.achitecture[layer + 1],1)
            self.deltaw[layer] = Matrix(1,1)
            self.deltab[layer] = Matrix(1,1)

    def inputData(self , listOfInputs):
        for j in range(len(listOfInputs)):
            self.inputTraining.append(Matrix(len(listOfInputs[j]),1))
            for i in range(len(listOfInputs[j])):
                self.inputTraining[j].matrix[i][0] = listOfInputs[j][i]


    def targetData(self , listOfTargets):
        for j in range(len(listOfTargets)):
            self.targetTraining.append(Matrix(len(listOfTargets[j]),1))
            for i in range(len(listOfTargets[j])):
                self.targetTraining[j].matrix[i][0] = listOfTargets[j][i]
 
    def feedForward(self,inputData):
        MatrixOfInputData = Matrix(self.achitecture[0],1)
        for i in range(self.achitecture[0]):
            MatrixOfInputData.matrix[i][0] = inputData[i]
        self.data[0] = MatrixOfInputData
        for layer in range(len(self.achitecture) - 1):
            self.datab[layer+1].matrix = add(multiply(self.w[layer].matrix,self.data[layer].matrix),self.bias[layer].matrix)
            self.data[layer+1].matrix = activationFunction(self.datab[layer + 1].matrix)
        return(self.data[-1].matrix)
    
    def train(self,NumberOfTraining):
        for i in range(NumberOfTraining):
            for numberOfTrainingData in range(len(self.inputTraining)):
                #feedForward
                self.data[0] = self.inputTraining[numberOfTrainingData]
                for layer in range(len(self.achitecture) - 1):
                    self.datab[layer+1].matrix = add(multiply(self.w[layer].matrix,self.data[layer].matrix),self.bias[layer].matrix)
                    self.data[layer+1].matrix = activationFunction(self.datab[layer + 1].matrix)
                #BackPropagation      
                self.error[-1].matrix = subtract(self.targetTraining[numberOfTrainingData].matrix,self.data[-1].matrix)
                for layer in range(len(self.achitecture) - 3,-1,-1):
                    self.error[layer].matrix = multiply(transpose(self.w[layer+1].matrix),self.error[layer+1].matrix)
                for layer in range(len(self.achitecture) - 1):
                    self.deltaw[layer].matrix = multiply(scalarMultiply(self.lr,sMultiply(self.error[layer].matrix,dSigmoid(self.data[layer+1].matrix))),transpose(self.data[layer].matrix))
                    self.deltab[layer].matrix = scalarMultiply(self.lr,sMultiply(self.error[layer].matrix,dSigmoid(self.data[layer+1].matrix)))
                    self.w[layer].matrix = add(self.w[layer].matrix,self.deltaw[layer].matrix)
                    self.bias[layer].matrix = add(self.bias[layer].matrix,self.deltab[layer].matrix)
    def percentCorrect(self,listofTestingInputData,listofTestingLabelData):
        count = 0
        for i in range(len(listofTestingInputData)):
            if(np.argmax(self.feedForward(listofTestingInputData[i])) == listofTestingLabelData[i] ):
                count += 1
        return((count / len(listofTestingLabelData)) * 100)

    def getWeight(self):
        w = []
        for i in range(len(self.achitecture) - 1):
            w.append(self.w[i].matrix)
        return(w)

    def getBias(self):
        b = []
        for i in range(len(self.achitecture) - 1):
            b.append(self.bias[i].matrix)
        return(b)
    
    def getWeightAndBias(self):
        k = []
        k.append(self.getWeight())
        k.append(self.getBias())
        return(k)

    def saveWeight(self,fileName):
        np.save('%s_weight.npy'%fileName,self.getWeight())

    def saveBias(self,fileName):
        np.save('%s_bias.npy'%fileName,self.getBias())
    
    def saveWeightAndBias(self,fileName):
        np.save('%s_wb.npy'%fileName,self.getWeightAndBias())

    def injectWeight(self,listofweight):
        for i in range(len(listofweight)):
            self.w[i].matrix = listofweight[i]

    def injectBias(self,listofbias):
        for i in range(len(listofbias)):
            self.bias[i].matrix = listofbias[i]

    def injectWeightAndBias(self,listofWB):
        self.injectWeight(listofWB[0])
        self.injectBias(listofWB[1])

