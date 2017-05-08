import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
import timeit


start_time = timeit.default_timer()

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn, n)
	d = dict([(bigram, True) for bigram in bigrams])
	d.update(best_word_feats(words))
	return d
print 'evaluating best word features'

print 'Reading file...'
reviewFile = pd.read_csv('yelp_academic_dataset_review.csv', nrows = 70000, usecols = [3,5])

reviews = reviewFile.text
stars = reviewFile.stars

stop = set(stopwords.words('english'))
stemmer = LancasterStemmer()

i = 0
fivestars = []
fourstars = []
threestars = []
twostars = []
onestars = []
for review in reviews:
	if stars[i] == 5:
		fivestars.append(review)
	if stars[i] == 4:
		fourstars.append(review)
	if stars[i] == 3:
		threestars.append(review)
	if stars[i] == 2:
		twostars.append(review)
	if stars[i] == 1:
		onestars.append(review)
	i = i + 1 
 
word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

print 'Getting words...'
for review in fivestars:
	if type(review) is str:
		for word in review.split():
			if word not in stop:
				word_fd.update(stemmer.stem(word.decode('utf-8')).lower())
				label_word_fd['5'].update(stemmer.stem(word.decode('utf-8')).lower())
 
for review in fourstars:
	if type(review) is str:
		for word in review.split():
			word_fd.update(stemmer.stem(word.decode('utf-8')).lower())
			label_word_fd['4'].update(stemmer.stem(word.decode('utf-8')).lower())
    
for review in threestars:
	if type(review) is str:
		for word in review.split():
			word_fd.update(stemmer.stem(word.decode('utf-8')).lower())
			label_word_fd['3'].update(stemmer.stem(word.decode('utf-8')).lower())
 
for review in twostars:
	if type(review) is str:
		for word in review.split():
			word_fd.update(stemmer.stem(word.decode('utf-8')).lower())
			label_word_fd['2'].update(stemmer.stem(word.decode('utf-8')).lower())
			
for review in onestars:
	if type(review) is str:
		for word in review.split():
			word_fd.update(stemmer.stem(word.decode('utf-8')).lower())
			label_word_fd['1'].update(stemmer.stem(word.decode('utf-8')).lower())
 
five_word_count = label_word_fd['5'].N()
four_word_count = label_word_fd['4'].N()
three_word_count = label_word_fd['3'].N()
two_word_count = label_word_fd['2'].N()
one_word_count = label_word_fd['1'].N()
total_word_count = five_word_count + four_word_count + three_word_count + two_word_count + one_word_count

word_scores = {}

for word, freq in word_fd.iteritems():
    five_score = BigramAssocMeasures.chi_sq(label_word_fd['5'][word],
        (freq, five_word_count), total_word_count)
    four_score = BigramAssocMeasures.chi_sq(label_word_fd['4'][word],
        (freq, four_word_count), total_word_count)
    three_score = BigramAssocMeasures.chi_sq(label_word_fd['3'][word],
        (freq, three_word_count), total_word_count)
    two_score = BigramAssocMeasures.chi_sq(label_word_fd['2'][word],
        (freq, two_word_count), total_word_count)
    one_score = BigramAssocMeasures.chi_sq(label_word_fd['1'][word],
        (freq, one_word_count), total_word_count)
    word_scores[word] = five_score + four_score + three_score + two_score + one_score
 
best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])

print 'Getting features...'
fiveStarsFeats = [(best_bigram_word_feats(review.split()), '5') for review in fivestars if type(review) is str]
fourStarsFeats = [(best_bigram_word_feats(review.split()), '4') for review in fourstars if type(review) is str]
threeStarsFeats = [(best_bigram_word_feats(review.split()), '3') for review in threestars if type(review) is str]
twoStarsFeats = [(best_bigram_word_feats(review.split()), '2') for review in twostars if type(review) is str]
oneStarsFeats = [(best_bigram_word_feats(review.split()), '1') for review in onestars if type(review) is str]
 
fivecutoff = len(fiveStarsFeats)*3/4
fourcutoff = len(fourStarsFeats)*3/4
threecutoff = len(threeStarsFeats)*3/4
twocutoff = len(twoStarsFeats)*3/4
onecutoff = len(oneStarsFeats)*3/4

trainfeats = fiveStarsFeats[:fivecutoff] + fourStarsFeats[:fourcutoff] + threeStarsFeats[:threecutoff] + twoStarsFeats[:twocutoff] + oneStarsFeats[:onecutoff]
testfeats = fiveStarsFeats[fivecutoff:] + fourStarsFeats[fourcutoff:] + threeStarsFeats[threecutoff:] + twoStarsFeats[twocutoff:] + oneStarsFeats[onecutoff:]

print 'Training model...'
classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)

print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)

classifier.show_most_informative_features()
print 'Elapsed Time ' + str(timeit.default_timer() - start_time)
