import sklearn
import nltk
import csv
import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from __future__ import print_function

def GetData():
	with open('train_set_x.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		trainData = []
		for row in readTest:
			trainData.append(row[1])
		
		trainData.pop(0)
	return trainData

def GetData2():
	with open('test_set_x.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		trainData = []
		for row in readTest:
			trainData.append(row[1])
		
		trainData.pop(0)
	return trainData

def GetTarget():
	with open('train_set_y.csv') as csvfile:
		readTest = csv.reader(csvfile, delimiter=',')
		testTarget = []
		for row in readTest:
			testTarget.append(row[1])
		testTarget.pop(0)
	return testTarget

probs={}
def proportions(c):
    if c not in probs:
        i=0
        numLetter=[0,0,0,0,0]
        for sentence in trainDataLetterized:
            for letter in sentence:
                if letter==c:
                    numLetter[int(testTarget[i])]+=1
            i+=1
        probLetter=[]
        i=0
        for num in numLetter:
            probLetter.append(num/float(numLang[i]))
            i+=1
        probs[c]=probLetter
    return probs[c]

trainData=GetData()
trainDataLetterized=[]
for sentence in trainData:
    trainDataLetterized.append(list(sentence))

testTarget=GetTarget()

numLang=[0,0,0,0,0]
otherNumLang=[0,0,0,0,0]
i=0
for sentence in trainDataLetterized:
    otherNumLang[int(testTarget[i])]+=1
    for letter in sentence:
        if letter!=' ':
            numLang[int(testTarget[i])]+=1
    i+=1

testData=GetData2()
testDataLetterized=[]
for sentence in testData:
    testDataLetterized.append(list(sentence))
total=len(trainData)    
frequency=[]
for num in otherNumLang:
    frequency.append(num/float(total))    
guess=[]
for sentence in testDataLetterized:
    product=[1,1,1,1,1]
    for character in sentence:
        if character != ' ':
            for i in range(5):
                product[i]*=proportions(character)[i]
    maxIndex=0
    for i in range(5):
        product[i]*=frequency[i]
    for i in range(5):
        if product[i]>product[maxIndex]:
                maxIndex=i
    guess.append(maxIndex)

with open("theoutput.csv",'wb') as predictions:
		wr = csv.writer(predictions, delimiter=',')
		wr.writerow(['Id','Category'])
		for idx, value in enumerate(guess):
			wr.writerow([idx, value])