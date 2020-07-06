import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np


# func to train a GCN
def trainGCN(data, labels):
  rows, cols = data.shape

  # initialize the W and Z matrix
  W = (np.random.random((30, 30)) - 0.5) / 10
  Z = (np.random.random((30, 1)) - 0.5) / 10

  # initialize total iterations & learning rate
  iterations = 1000
  lr = 0.5

  for i in range(iterations):
    # Forward pass      
    X1 = np.dot(data, W)
    X11 = 1 / (1 + np.exp(-(X1)))
    X2 = np.dot(X11, Z)
    X22 = 1 / (1 + np.exp(-(X2)))

    # Backprop 
    L = (labels - X22) / rows
    L1 = L * (X22 * (1 - X22))
    L11 = L1.dot(Z.T)
    L2 = L11 * (X11 * (1 - X11))

    # Weight update
    Z += (lr * (X11.T.dot(L1)))
    W += (lr * (data.T.dot(L2)))

  return W, Z


# find the euclidian dist between two vectors
def findDist(vec1, vec2):
  dist = 0.0
  size = len(vec1)

  for i in range(size):
    dist += (vec1[i] - vec2[i])**2

  return math.sqrt(dist)


# find neighbourhood aggregation for given data points amongest them
def findNeighbourAggregation(data):
  # using 5 nearest neighbour + one self 
  # denominatior = sqrt(|N(v)|*|N(u)|) = 5
  # numerator = summation of neighbours

  totalRows, totalCols = data.shape
  finalAgg = []

  for i in range(totalRows):

    row1 = data[i]
    dist = [] 
    idx  = [] # store the 5 index which are nearest to the curr index

    for j in range(totalRows):

      if (i == j):
        continue

      d = findDist(row1, data[j])

      if (len(dist) < 5):
        dist.append(d)
        idx.append(j)
      elif(d < max(dist)):
        maxIdx = dist.index(max(dist))
        dist[maxIdx] = d
        idx[maxIdx]  = j

    # summing over the neighbour for curr index
    for k in range(5):
      row1 += data[idx[k]]

    # dividing by the denominator
    row1 = row1 / 5.0

    finalAgg.append(row1)

  return np.asarray(finalAgg)


# find neighbourhood aggregation for a particular node in the data
def neighAggForNode(data, node):
  # using 5 nearest neighbour + one self 
  # denominatior = sqrt(|N(v)|*|N(u)|) = 5
  # numerator = summation of neighbours

  totalRows, totalCols = data.shape
  finalAgg = []
  
  row1 = node
  dist = [] 
  idx  = [] # store the 5 index which are nearest to the curr index

  for i in range(totalRows):
    d = findDist(row1, data[i])

    # here data already contains the node so computing 6 (5 neigh + 1 self)
    if (len(dist) < 6):
      dist.append(d)
      idx.append(i)
    elif(d < max(dist)):
      maxIdx = dist.index(max(dist))
      dist[maxIdx] = d
      idx[maxIdx]  = i

  agg = np.zeros((1, totalCols))
  # summing over the neighbour for curr index
  for j in range(5):
    agg += data[idx[j]]

  # dividing by the denominator
  agg = agg / 5.0

  return np.asarray(agg)


def doAggregation(trainData, newData):
  totalRows, totalCols = newData.shape
  finalAgg = []

  for i in range(totalRows):
    agg = neighAggForNode(trainData, newData[i])
    finalAgg.append(agg)

  return np.asarray(finalAgg)


# if prob > 0.5 treat as class 1 else as class 0
def interpretResult(result):
  size = len(result)
  newResult = np.ones((size,1), int)
  for i in range(size):
    if (result[i] > 0.5):
      newResult[i] = 1
    else:
      newResult[i] = 0

  return newResult


def computeAcc(W, Z, data, labels):
  X1 = np.dot(data, W)
  X11 = 1 / (1 + np.exp(-(X1)))
  X2 = np.dot(X11, Z)
  X22 = 1 / (1 + np.exp(-(X2)))

  return metrics.accuracy_score(interpretResult(X22), labels)

  
def interpretLabel(lbl):
  size = len(lbl)
  newLabel = c = np.ones((size,1), int)
  for i in range(size):
    if (lbl[i] == 'M'):
      newLabel[i] = 1
    else:
      newLabel[i] = 0

  return newLabel


# load the pre-processed data
train = pd.read_csv('data/train.data', header=None)
val   = pd.read_csv('data/val.data', header=None)
test  = pd.read_csv('data/test.data', header=None)

trainFeatures = train.drop([0], axis=1)
trainLabels   = train[0]

valFeatures = val.drop([0], axis=1)
valLabels   = val[0]

testFeatures = test.drop([0], axis=1)
testLabels   = test[0]

totalFeaturesVec = np.concatenate((trainFeatures.values, valFeatures.values, testFeatures.values), axis=0)

# performing the neighbourhood aggregation
aggTrainFeatures = findNeighbourAggregation(trainFeatures.values)
aggValFeatures   = doAggregation(totalFeaturesVec, valFeatures.values)
aggTestFeatures  = doAggregation(totalFeaturesVec, testFeatures.values)

W, Z = trainGCN(aggTrainFeatures, interpretLabel(trainLabels.values))

trainAcc = computeAcc(W, Z, aggTrainFeatures, interpretLabel(trainLabels.values))
valAcc   = computeAcc(W, Z, aggValFeatures, interpretLabel(valLabels.values))
testAcc  = computeAcc(W, Z, aggTestFeatures, interpretLabel(testLabels.values))

print('----------GCN---------')
print('Train Accuracy : ', trainAcc)
print('Val   Accuracy : ', valAcc)
print('Test  Accuracy : ', testAcc)
