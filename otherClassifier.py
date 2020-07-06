import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neighbors import KNeighborsClassifier

# load the pre-rocessed data
train = pd.read_csv('data/train.data', header=None)
val   = pd.read_csv('data/val.data', header=None)
test  = pd.read_csv('data/test.data', header=None)

trainFeatures = train.drop([0], axis=1)
trainLabels   = train[0]

valFeatures = val.drop([0], axis=1)
valLabels   = val[0]

testFeatures = test.drop([0], axis=1)
testLabels   = test[0]

#----------KNN----------
print("\n\n----------KNN----------")
knnClassifier = KNeighborsClassifier()
knnClassifier.fit(trainFeatures, trainLabels)

knnTrainPredictions = knnClassifier.predict(trainFeatures)
knnTrainAccuracy = metrics.accuracy_score(knnTrainPredictions, trainLabels)
print('Train Accuracy : ', knnTrainAccuracy)

knnValPredictions = knnClassifier.predict(valFeatures)
knnValAccuracy = metrics.accuracy_score(knnValPredictions, valLabels)
print('Val Accuracy   : ', knnValAccuracy)

knnTestPredictions = knnClassifier.predict(testFeatures)
knnTestAccuracy = metrics.accuracy_score(knnTestPredictions, testLabels)
print('Test Accuracy  : ', knnTestAccuracy)

#----------SVM----------
print("\n\n----------SVM----------")
svmClassifier = svm.LinearSVC()
#svmClassifier = svm.SVC()
svmClassifier.fit(trainFeatures, trainLabels)

svmTrainPredictions = svmClassifier.predict(trainFeatures)
svmTrainAccuracy = metrics.accuracy_score(svmTrainPredictions, trainLabels)
print('Train Accuracy : ', svmTrainAccuracy)

svmValPredictions = svmClassifier.predict(valFeatures)
svmValAccuracy = metrics.accuracy_score(svmValPredictions, valLabels)
print('Val Accuracy   : ', svmValAccuracy)

svmTestPredictions = svmClassifier.predict(testFeatures)
svmTestAccuracy = metrics.accuracy_score(svmTestPredictions, testLabels)
print('Test Accuracy  : ', svmTestAccuracy)

#----------Logistic Regression----------
print("\n\n----------Logistic Regression----------")
lrClassifier = linear_model.LogisticRegression()
lrClassifier.fit(trainFeatures, trainLabels)

lrTrainPredictions = lrClassifier.predict(trainFeatures)
lrTrainAccuracy = metrics.accuracy_score(lrTrainPredictions, trainLabels)
print('Train Accuracy : ', lrTrainAccuracy)

lrValPredictions = lrClassifier.predict(valFeatures)
lrValAccuracy = metrics.accuracy_score(lrValPredictions, valLabels)
print('Val Accuracy   : ', lrValAccuracy)

lrTestPredictions = lrClassifier.predict(testFeatures)
lrTestAccuracy = metrics.accuracy_score(lrTestPredictions, testLabels)
print('Test Accuracy  : ', lrTestAccuracy)
