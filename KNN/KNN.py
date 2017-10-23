# -*- coding: utf-8 -*-

import sklearn
import nltk
import unicodecsv as csv
import re
import numpy as np
import math
import operator
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# import pickle
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

# train_x_file = 'sampleTrain_x.csv'
# train_y_file = 'sampleTrain_y.csv'
# test_x_file = 'sampleTest_x.csv'
#
train_x_file = 'train_set_x.csv'
train_y_file = 'train_set_y.csv'
test_x_file = 'test_set_x.csv'

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))

def getData():
    with open(train_x_file) as csvfile:
        readTest = csv.reader(csvfile, delimiter=',')
        trainData = []
        for row in readTest:
            trainData.append(row[1])

        trainData.pop(0)
    return trainData

def getTarget():
    with open(train_y_file) as csvfile:
        readTest = csv.reader(csvfile, delimiter=',')
        testTarget = []
        for row in readTest:
            testTarget.append(row[1])
        testTarget.pop(0)
    return testTarget


def getTest():
    with open(test_x_file) as csvfile:
        readTest = csv.reader(csvfile, delimiter=',')
        testData = []
        for row in readTest:
            testData.append(row[1])
        testData.pop(0)
    return testData

def dataToVec(traindata, testdata):
    count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
    X_train_counts = count_vect.fit_transform(traindata)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_test_counts = count_vect.transform(testdata)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    return X_train_tfidf, X_test_tfidf

def validation(traindata):
    count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
    X_train_counts = count_vect.fit_transform(traindata[:20000])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf

# def testDataToVec(data):
    # count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
    # X_test_counts = count_vect.transform(data)
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # return X_train_tfidf


def hashAndTrain():
	xp = getData()
	y = getTarget()
	t = getTest()
	x = []
	for idx, lines in enumerate(xp):
		# Remove white spaces since we're working with unigram
		lines = ((re.sub(r'\s+', '', lines)).replace("", " ")[1: -1]).split(" ")
		shuffle(lines)
		lines = " ".join(lines)[:41].lower()
		#lines = " ".join(lines).lower()
		x.append(lines)

	trainSize = len(x)*100/100
	cvSize    = len(x)*0/100
	
	xCrossVal = x[0:(cvSize)]
	xTrain 	  = x[(cvSize):(cvSize+trainSize)]
	
	yCrossVal = y[0:(cvSize)]
	yTrain 	  = y[(cvSize):(cvSize+trainSize)]
		
	count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, analyzer='char_wb', ngram_range=(1, 1))
	tfidf_transformer = TfidfTransformer(sublinear_tf=True)
	X_train_counts = count_vect.fit_transform(xTrain)
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	# Create a random binary hash with 10 bits
	rbp = RandomBinaryProjections('rbp', 10)
	m , n = X_train_counts.shape
	engine = Engine(n, lshashes=[rbp])
	
	for i in range(m):
		entry = np.transpose(X_train_tfidf[i])
		engine.store_vector(entry, i)
		if (i % 1000) == 0:
			print "Creating the engine: We are {}% complete".format(i*100/float(m))
	
	#Cross Validate only if there's a CV data
	if cvSize > 0:
		X_cv_counts = count_vect.transform(xCrossVal)
		X_cv_tfidf = tfidf_transformer.transform(X_cv_counts)
		
		CV_nearest = []
		m , n = X_cv_counts.shape	
		for i in range(m):
			N = engine.neighbours(np.transpose(X_cv_tfidf[i]))
			CV_nearest.append(N)
			if (i % 100) == 0:
				print "Finding K-nearest neighborus: We are {}% complete".format(i*100/float(m))
		
		K_List = [1,2,3,5,8,13]
		for K in K_List:	
			K_nearest = []
			for i in range(len(CV_nearest)):
				N = CV_nearest[i]
				if len(N) > K:
					N = N[0:K]
				K_nearest.append(N)
				
			result = []
			for i in range(len(K_nearest)):
				prediction = []
				for j in range(len(K_nearest[i])):
					prediction.append(yTrain[K_nearest[i][j][1]])
				result.append(prediction)
			
			predicted = []
			for i in range(len(result)):
				if result[i]:
					predicted.append(str(np.argmax(np.bincount(result[i]))))
				else:
					# Predict french if no result is associated
					predicted.append(u'1')
			
			accuracy = sklearn.metrics.accuracy_score(yCrossVal, predicted)
			print "{a}% accuracy using {k} nearest neighbours".format(a = accuracy*100, k=K)
	
	t = getTest()
	X_test_counts = count_vect.transform(t)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	
	Test_nearest = []
	m , n = X_test_counts.shape	
	for i in range(m):
		N = engine.neighbours(np.transpose(X_test_tfidf[i]))
		Test_nearest.append(N)
		if (i % 100) == 0:
			print "Finding K-nearest neighborus: We are {}% complete".format(i*100/float(m))
	
	K_List = [1,2,3,5,8,13]
	for K in K_List:	
		K_nearest = []
		for i in range(len(Test_nearest)):
			N = Test_nearest[i]
			if len(N) > K:
				N = N[0:K]
			K_nearest.append(N)
			
		result = []
		for i in range(len(K_nearest)):
			prediction = []
			for j in range(len(K_nearest[i])):
				prediction.append(yTrain[K_nearest[i][j][1]])
			result.append(prediction)
		
		predicted = []
		for i in range(len(result)):
			if result[i]:
				predicted.append(str(np.argmax(np.bincount(result[i]))))
			else:
				# Predict french if no result is associated
				predicted.append(u'1')
		
		with open('myPredictionKNN_{m}.csv'.format(m=K), 'wb') as predictions:
			wr = csv.writer(predictions, delimiter=',')
			wr.writerow(['Id','Category'])
			for idx, value in enumerate(predicted):
				wr.writerow([idx, value])

	
	return Test_nearest, result, yTrain, predicted, engine
	
def weighted_result(CV_nearest,yTrain):
	weights = [0,0,0,0,0]
	for j in range(len(CV_nearest)):
		value = yTrain[CV_nearest[j][1]]
		distance = CV_nearest[j][2]
		if distance == 0.0:
			return value
		else:	
			weights[int(value)] = 1/float(distance)
	return u'{}'.format(str(np.argmax(weights)))


# hashAndTrain()
def main():
    train_data, test_data = dataToVec(getData(), getTest())
    train_data_y = getTarget()

    train_data = train_data.toarray().tolist()
    test_data = test_data.toarray().tolist()
    for i in range(len(train_data)):
        train_data[i].extend(train_data_y[i])
    #
    predictions = []
    k = 3
    f = open('result.csv', 'w')
    f.write('Id,Category\n')
    for x in range(len(test_data)):
        neighbors = getNeighbors(train_data, test_data[x], k)
        result = getResponse(neighbors)
        f.write(str(x) + ',' + result + '\n')
        predictions.append(result)
        print '> point ' + repr(x) + ' prediction=' + repr(result)

	# neigh = KNeighborsClassifier(n_neighbors=9)
	# neigh.fit(train_data, train_data_y)
	# prediction = neigh.predict(test_data)
	#
	# for x in range(len(test_data)):
	#     print '> point ' + repr(x) + ' prediction=' + repr(prediction[x]) + ', actual=' + repr(test_data[x][-1])

	#
	# for i in range(len(train_data)):
	#     train_data[i].extend(train_data_y[i])


	# predictions = []
	# k = 3
	# for x in range(len(test_data)):
	#     neighbors = getNeighbors(train_data, test_data[x], k)
	#     result = getResponse(neighbors)
	#     predictions.append(result)
	#     print '> point ' + repr(x) + ' prediction=' + repr(result) + ', actual=' + repr(test_data[x][ -1])
	# accuracy = getAccuracy(test_data, predictions)
	# print repr(accuracy)



	## Using sklearn

	# neigh = KNeighborsClassifier(n_neighbors=9)
	# neigh.fit(train_data, train_data_y)
	#
	# with open('train_data.txt', 'wb') as fp:
	#     pickle.dump(train_data, fp)
	# with open('test_data.txt', 'wb') as fp1:
	#     pickle.dump(test_data, fp1)
	# with open('fit_data.txt', 'wb') as fp2:
	#     pickle.dump(neigh, fp2)
	# exit(0)

	# with open('train_data.txt', 'rb') as fp:
	#     train_data = pickle.load(fp)
	# with open('test_data.txt','rb') as fp1:
	#     test_data = pickle.load(fp1)
	# with open('fit_data.txt', 'rb') as fp2:
	#     neigh = pickle.load(fp2)
	#
	# print 'Finish loading data.'
	# predicted = neigh.predict(test_data[:5000])
	# print 'Finish prediction.'
	# #
	# with open('result.csv', 'w') as f:
	#     # f.write('Id,Category\n')
	#     for i in range(len(predicted)):
	#         f.write(str(i)+','+str(predicted[i])+'\n')


	# print np.shape(test_data)
	# print np.shape(train_data)

	#
	# for i in range(len(train_data)):
	#     train_data[i].extend(train_data_y[i])
	#
	# length = len(train_data[0]) - len(test_data[0]) - 1
	#
	# for i in range(len(test_data)):
	#     test_data[i].extend([0]*length)
	#     # test_data[i].extend('-1')
	#
	# print np.shape(test_data)
	# print np.shape(train_data)
	#
	# predictions = []
	# k = 3
	# for x in range(len(test_data[:500])):
	#     neighbors = getNeighbors(train_data, test_data[x], k)
	#     result = getResponse(neighbors)
	#     predictions.append(result)
	#     print '> point ' + repr(x) + ' prediction=' + repr(result)

	# test_size = 100
	# test_data = train_data[-test_size:]
	# train_data = train_data[:-test_size]
	# print np.shape(train_data), np.shape(test_data)
	#
	# predictions = []
	# k = 9
	# for x in range(len(test_data)):
	#     neighbors = getNeighbors(train_data, test_data[x], k)
	#     result = getResponse(neighbors)
	#     predictions.append(result)
	#     print '> point ' + repr(x) + ' prediction=' + repr(result) + ', actual=' + repr(test_data[x][ -1])
	# accuracy = getAccuracy(test_data, predictions)
	# print repr(accuracy)


main()