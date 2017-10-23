# -*- coding: utf-8 -*-
import os.path
import sklearn
import nltk
import unicodecsv as csv
import re
import numpy
import string
from langdetect import detect_langs
from random import shuffle
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier

def GetCleanData():
	with open('myCleanData.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		trainData = []
		for row in readTest:
			trainData.extend((row[0],row[1],row[2]))
		trainData = trainData[3:]
	return trainData

def GetData():
	with open('../Data/train_set_x.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		trainData = []
		for row in readTest:
			trainData.append(row[1])
		trainData.pop(0)
	return trainData

def GetTarget():
	with open('../Data/train_set_y.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		testTarget = []
		for row in readTest:
			testTarget.append(row[1])
		testTarget.pop(0)
	return testTarget

def GetTest():
	with open('../Data/test_set_x.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		testData = []
		for row in readTest:
			testData.append(row[1])
		testData.pop(0)
	return testData

def CleanData(check_lang):
	x = GetData()
	y = GetTarget()
	xy = []
	total_dropped = [0,0,0,0,0];
	for idx, lines in enumerate(x):
		# Get rid of all the words containing a digit 
		lines = re.sub(r'\w*\d\w*', '', lines).strip()
		# Get rid of all punctuation
		# lines = lines.translate(string.maketrans("",""), string.punctuation).strip()
		# Remove words longer than 15 characters
		lines = re.sub(r'\b\w{14,}\b', '', lines).strip()
		# Get rid of bad training data by checking language 
		# Also make sure it's properly labelled 
		if check_lang:
			if lines:
				try:
					lang = getLanguage(lines)
				except:
					lang = u'na'
				if lang != y[idx]:
					#print "{i}. Detected language is {d} and should be {l}".format(i=idx, d=lang, l=y[idx])
					lines = []
					total_dropped[int(y[idx])] += 1
			if (idx % 1000) == 0:
				print idx
		
		if lines:
			# Remove white spaces since we're working with unigram
			lines = ((re.sub(r'\s+', '', lines)).replace("", " ")[1: -1]).split(" ")
			shuffle(lines)
			lines = " ".join(lines)[:41].lower()
			#lines = " ".join(lines).lower()
			xy.append(idx)
			xy.append(y[idx])
			xy.append(lines)
	
	if check_lang:
		print "Dropped this many packages due to language-mismatch {}".format(total_dropped)
		
	return xy

def getLanguage(x):
	y = detect_langs(x)
	result = u'na'
	if 'sk' in str(y) or 'sl' in str(y):
		result = u'0'
	elif 'fr' in str(y):
		result = u'1'
	elif 'es' in str(y):
		result = u'2'
	elif 'de' in str(y):
		result = u'3'
	elif 'pl' in str(y):
		result = u'4'			
	return result
	
def CreateCleanData():
	cleanData = CleanData(0)
	x = cleanData[1::3]
	y = cleanData[2::3]
	
	with open('myCleanData.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category','Input'])
		for idx, line in enumerate(x):
			wr.writerow([idx, y[idx], line ])

def LimitTrainData(x, y):
	# Normalize sizes. Take 5000 of each language	
	finalX = []
	finalY = []
	for i in numpy.unique(y):
		tempX = []
		for idx, value in enumerate(y):
			if y[idx] == i:
				tempX.append(x[idx])
		shuffle(tempX)
		for index in range(0,5000):
			finalX.append(tempX[index])
			finalY.append(i)
	x = finalX;
	y = finalY;
	return [x, y]

def GetTrainDataAndSplit():
	cleanData = GetCleanData()
	x = cleanData[2::3]
	y = cleanData[1::3]
	
	with open('myCleanData.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category','Input'])
		for idx, line in enumerate(x):
			wr.writerow([idx, y[idx], line ])
	
	trainSize = len(x)*80/100
	cvSize    = len(x)*20/100
	
	xCrossVal = x[0:(cvSize)]
	xTrain 	  = x[(cvSize):]
	
	yCrossVal = y[0:(cvSize)]
	yTrain 	  = y[(cvSize):]
	yTest 	  =[]
	
	count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
	tfidf_transformer = TfidfTransformer(sublinear_tf=True)
	#X_counts = count_vect.fit_transform(GetTest())[:,2:]
	X_train_counts = count_vect.fit_transform(xTrain)[:,1:]
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	X_cv_counts = count_vect.transform(xCrossVal)[:,1:]
	X_cv_tfidf = tfidf_transformer.transform(X_cv_counts)

	X_test_counts = count_vect.transform(GetTest())[:,1:]
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	
	return X_train_counts, yTrain, X_cv_counts, yCrossVal, X_test_counts, yTest

def TrainDataWithSVM():
	X_train_tfidf, yTrain, X_cv_tfidf, yCrossVal, X_test_tfidf, yTest = GetTrainDataAndSplit()
	methods = ['log', 'hinge', 'modified_huber',]
	penalties = ['none', 'l2', 'elasticnet']
	results = []
	bestAccuracy = 0
	for method in methods:
		for pen in penalties:
			clf = SGDClassifier(loss=method, penalty=pen, alpha=1e-5, random_state=314)
			clf.fit(X_train_tfidf,yTrain)
			predicted = clf.predict(X_cv_tfidf)
			accuracy = sklearn.metrics.accuracy_score(yCrossVal,predicted)
			
			if accuracy > bestAccuracy:
				bestAccuracy = accuracy
				bestM = method
				bestP = pen
				
			print "Training SVM with {m} method with {p} penalty yielded {r}".format(m=method,p = pen, r=accuracy)
			results.extend((method,pen,accuracy))
	
	clf = SGDClassifier(loss=bestM, penalty=bestP, alpha=1e-5, random_state=314)
	clf.fit(X_train_tfidf,yTrain)								
	predicted = clf.predict(X_test_tfidf);		
	with open('myPredictionNN_{m}_{p}.csv'.format(m=method, p=pen), 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category'])
		for idx, value in enumerate(predicted):
			wr.writerow([idx, value])
			
	ms = results[0::3]
	ps = results[1::3]
	accuracies = results[2::3]
	
	with open('mySVMtesting.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Idx','Method','Penalty', 'Accuracy'])
		for idx, value in enumerate(ms):
			wr.writerow([idx, value, ps[idx], accuracies[idx]])
			
	return 0

def TrainDataWithABC(verbose):
	cleanData = GetCleanData()
	x = cleanData[2::3]
	y = cleanData[1::3]

	if verbose:
		trainSize = len(x)*80/100
		cvSize    = len(x)*20/100
		
		xCrossVal = x[0:(cvSize)]
		xTrain 	  = x[(cvSize):]
		
		yCrossVal = y[0:(cvSize)]
		yTrain 	  = y[(cvSize):]
	else:
		xTrain 	  = x
		yTrain	  = y
	
	count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1, 1))
	X_train_counts = count_vect.fit_transform(xTrain)[:,2:]
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
	clf.fit(X_train_tfidf,yTrain)
	
	if verbose:
		X_cv_counts = count_vect.transform(xCrossVal)[:,2:]
		X_cv_tfidf = tfidf_transformer.transform(X_cv_counts)
		predicted = clf.predict(X_cv_tfidf)
		accuracy = sklearn.metrics.accuracy_score(yCrossVal,predicted)
		print "Training ABC yielded {r}".format(r=accuracy)
		probs = clf.predict_proba(X_cv_tfidf)

	return clf
	
def TrainDataWithABCandNN(verbose):
	cleanData = GetCleanData()
	xp = cleanData[2::3]
	y = cleanData[1::3]
	t = GetTest()
	testSize  =  len(t)
	
	x = []
	for idx, lines in enumerate(xp):
		# Remove white spaces since we're working with unigram
		lines = ((re.sub(r'\s+', '', lines)).replace("", " ")[1: -1]).split(" ")
		shuffle(lines)
		lines = " ".join(lines)[:41].lower()
		#lines = " ".join(lines).lower()
		x.append(lines)
	
	# alphabets = GetAlphabet()
	# Slovak = ['aáäbcčdďdzdžeéfghchiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž']
	# French = ['abcdefghijklmnopqrstuvwxyzéèçëòôöùàâ']
	# German = ['abcdefghijklmnopqrstuvwxyzäöüß']
	# Spanish = ['abcdefghijklmnñopqrstuvwxyzáéíóúñÑüÜ¿']
	# Polish = ['abcdefghijklmnopqrstuvwxyząćęłńóśźż']
	# alphabets.extend((Slovak,French,German,Spanish))
	
	count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, analyzer='char_wb', ngram_range=(1, 1))
	tfidf_transformer = TfidfTransformer(sublinear_tf=True)
	
	train_set_vectorization = count_vect.fit_transform(x)[:,1:]
	train_set_tfidf = tfidf_transformer.fit_transform(train_set_vectorization)
	
	#X_train_counts = count_vect.fit_transform(x)[:,2:]
	#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	X_train_counts = count_vect.transform(x[0:-40000])[:,1:]
	X_train_tfidf = tfidf_transformer.transform(X_train_counts)
	
	X_test_counts = count_vect.transform(t)[:,1:]
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	
	X_cv_counts = count_vect.transform(x[-40000:])[:,1:]
	X_cv_tfidf = tfidf_transformer.transform(X_cv_counts)

	ab_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
	ab_clf.fit(X_train_tfidf,y[0:-40000])
	nn_clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, 5))
	nn_clf.fit(X_train_tfidf, y[0:-40000])
	
	result = ab_clf.predict(X_cv_tfidf)
	accuracy = sklearn.metrics.accuracy_score(y[-40000:],result)
	print "{}% was classified using AdaBoost".format(accuracy*100)
	
	result = nn_clf.predict(X_cv_tfidf)
	accuracy = sklearn.metrics.accuracy_score(y[-40000:],result)
	print "{}% was classified using NN".format(accuracy*100)
	
	# Check the ones properly classified by AdaBoostClassifier
	probs = ab_clf.predict_proba(X_test_tfidf)
	result = ab_clf.predict(X_test_tfidf)
	classified = []
	toBeClassified = []
	for idx, input in enumerate(probs):
		if max(input) > 0.24:
			classified.extend((idx, result[idx]))
		else:
			toBeClassified.extend((idx, t[idx]))
			
	print "{}% was classified using AdaBoost".format(len(classified)*50/float(testSize))
	t = toBeClassified[1::2]
	indices = toBeClassified[0::2]
	
	X_test_counts = count_vect.transform(t)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	predicted = nn_clf.predict(X_test_tfidf);
								
	with open('myPredictionAB_NN.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category'])
		for idx in range (0, testSize):
			if classified and (idx == classified[0]):
				value = classified[1]
				classified = classified[2:]
			elif indices and (idx == indices[0]):
				value = predicted[0]
				indices = indices[1:]
				predicted = predicted[1:]
			wr.writerow([idx, value])
	
	# Now training purely NN
	print "Now training purely with neural net" 
	
	X_train_counts = count_vect.transform(x)
	X_train_tfidf = tfidf_transformer.transform(X_train_counts)
	X_test_counts = count_vect.transform(GetTest())
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	
	nn_clf2 = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(100, 5))
	nn_clf2.fit(X_train_tfidf, y)
	predicted = nn_clf2.predict(X_test_tfidf)
								
	with open('myPredictionNN.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category'])
		for idx, value in enumerate(predicted):
			wr.writerow([idx, value])
	
	return nn_clf
	
def TrainDataWithNN(cv_enabled = 0):
	print "Loading training data and applying pre-processing"
	
	if os.path.isfile('myCleanData.csv'):
		cleanData = GetCleanData()
		x = cleanData[2::3]
		y = cleanData[1::3]
	else:
		cleanData = CleanData(0)
		x = cleanData[2::3]
		y = cleanData[1::3]
		with open('myCleanData.csv', 'wb') as predictions:
			wr = csv.writer(predictions, delimiter=',')
			wr.writerow(['Id','Category','Input'])
			for idx, line in enumerate(x):
				wr.writerow([idx, y[idx], line ])
		
	print "Data loaded. Vectorizing as mono-gram (n-gram =1) and applying TF-IDF"
	trainSize = len(x)*80/100
	cvSize    = len(x)*20/100
	
	xCrossVal = x[0:(cvSize)]
	xTrain 	  = x[(cvSize):]
	
	yCrossVal = y[0:(cvSize)]
	yTrain 	  = y[(cvSize):]
	
	count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, analyzer='char_wb', ngram_range=(1, 1))
	tfidf_transformer = TfidfTransformer(sublinear_tf=True)
	
	X_train_counts = count_vect.fit_transform(xTrain)[:,1:]
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	print "Applying 80-20 split for cross-validation and vectorizing test set as well!"
	X_cv_counts = count_vect.transform(xCrossVal)[:,1:]
	X_cv_tfidf = tfidf_transformer.transform(X_cv_counts)

	X_test_counts = count_vect.transform(GetTest())[:,1:]
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	
	if cv_enabled:
		print "Cross-Validation enabled. Will search for best hyperparameters" 
		# first_layer = [100, 200]
		# second_layer = [10, 20]
		# alpha = [1e-3,1e-5]
		# methods = ['adam','sgd', 'lbfgs']
		
		first_layer = [100]
		second_layer = [10, 20]
		alpha = [1e-3]
		methods = ['adam']
		
		results = []
		bestAccuracy = 0
		bestF = 20
		bestS = 10
		for method in methods:
			for first in first_layer:
				for second in second_layer:
					for al in alpha:
						if first > second:
							print "Cross-Validation: Training NN using {m} method with alpha of {a} and hidden layer size ({f},{s})".format(m=method,f = first,s = second,a=al)
							clf = MLPClassifier(solver=method, alpha=al, hidden_layer_sizes=(first, second))
							clf.fit(X_train_tfidf, yTrain)
							predicted = clf.predict(X_cv_tfidf)
							accuracy = sklearn.metrics.accuracy_score(yCrossVal,predicted)
							print "Cross-Validation: Training {m} with {f} and {s} yielded {r}".format(m=method,f = first,s = second,r=accuracy)
							results.extend((first,second,accuracy))
							if accuracy > bestAccuracy:
								bestAccuracy = accuracy
								bestMethod = method
								bestF = first
								bestS = second
								bestAlpha = al
								
							predicted = clf.predict(X_test_tfidf)								
							with open('myPredictionNN_{m}_{f}_{s}_{a}.csv'.format(m=method, f=first, s=second, a=al), 'wb') as predictions:
								wr = csv.writer(predictions, delimiter=',')
								wr.writerow(['Id','Category'])
								for idx, value in enumerate(predicted):
									wr.writerow([idx, value])
							
		firsts = results[0::3]
		seconds = results[1::3]
		accuracies = results[2::3]
		
		with open('myNNtesting_summary.csv', 'wb') as predictions:
			wr = csv.writer(predictions, delimiter=',')
			wr.writerow(['# of First Layer','# of Second Layer', 'Accuracy'])
			for idx, value in enumerate(firsts):
				wr.writerow([idx, value, seconds[idx], accuracies[idx]])
	else:
		print "Cross-Validation disabled. Will use the best known hyperparameters" 
		bestF = 200
		bestS = 20
		bestAlpha = 1e-3
		bestMethod = 'adam'	
		
	
	print "Finally, training NN with the optimized / best-known hyperparamters" 
	print "Method = {}".format(bestMethod) 
	print "Learning rate = {}".format(bestAlpha)
	print "Hidden layer sizes = ({f},{s})".format(f=bestF,s=bestS)
	
	clf = MLPClassifier(solver=bestMethod, alpha=bestAlpha, hidden_layer_sizes=(bestF, bestS))
	clf.fit(X_train_tfidf, yTrain)
	predicted = clf.predict(X_test_tfidf)
	with open('myPredictionNN_Best.csv', 'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category'])
		for idx, value in enumerate(predicted):
			wr.writerow([idx, value])


	return clf
