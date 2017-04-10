import nltk
import pandas as pd
from itertools import groupby
from nltk.corpus import stopwords

reviewFile = pd.read_csv('yelp_academic_dataset_review.csv',usecols = [3,5])
#reviewFile = pd.read_csv('TestData.csv');
trainingData = reviewFile[:len(reviewFile)/2]
testingData = reviewFile[len(reviewFile)/2:]

trainingReviews = trainingData.text
trainingStars = trainingData.stars

testingReviews = testingData.text
testingStars = testingData.stars

classificationSpace = [1,2,3,4,5]
vocabulary = []

stop = set(stopwords.words('english'))

iterator = 0

documentClassification = {}
wordCountInClassification = {}

for classification in classificationSpace:
	wordCountInClassification[classification] = []

for review in trainingReviews:
	tokens = nltk.word_tokenize(review.decode('utf-8').lower())
	documentClassification[review] = trainingStars[iterator]
	for token in tokens:
		if token not in stop and token != '.' :
			vocabulary.append(token)
			wordCountInClassification[documentClassification[review]].append(token)
	iterator = iterator + 1

vocabulary.sort()
vocabularySize = len([len(list(group)) for key, group in groupby(vocabulary)])

correctPredictions = 0
incorrectPredictions = 0
i = len(reviewFile)/2

for review in testingReviews:
	maxProbability = 0
	classificationOfNewReview = 0
	for classification in classificationSpace:
		classificationProbability = 1
		words = nltk.word_tokenize(review.decode('utf-8').lower())
		for word in words:
			if word not in stop and word != '.':
				probabilityOfCurrentWord = float((wordCountInClassification[classification].count(word) + 1)) / float((len(wordCountInClassification[classification]) + vocabularySize))
				classificationProbability *= probabilityOfCurrentWord
				
		if classificationProbability > maxProbability:
			maxProbability = classificationProbability
			classificationOfNewReview = classification
#	print "Predicted classification " + str(classificationOfNewReview)
#	print "actual classification " + str(testingStars[i])
#	print

	if classificationOfNewReview == testingStars[i]:
		correctPredictions = correctPredictions + 1

	else:
		incorrectPredictions = incorrectPredictions + 1
	i = i + 1

print "Correct Predictions " + str(correctPredictions)
print "Incorrect predictions "+ str(incorrectPredictions)
