#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from datetime import date
from datetime import timedelta
from nltk import NaiveBayesClassifier
import nltk.classify
from TwitterSearch import *
import tweepy
import pickle
import math
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#outputs past numdays days in format yyyy-mm-dd *NOT* including todays date
def last_dates(numdays):
	return [date.today()-timedelta(days=x) for x in range(0, numdays)]

# queries twitter given the query
def grab_tweets(query):
	#consumer_key = 'RXZnGOT0HIMzlvluC49971qnW'
	consumer_key = 'uTAwmsVanLasrKPnPL2vMYWFe' 
	#consumer_secret = 'f6ZXMiirMw1b9a9QNSZYxhIXHAV4MoxOA4y16pmWfJV8zG0Gam' 
	consumer_secret = '9mZvzzsdbv1Hf1RzrOd1cMRTn2PjivqE4XWgzr9mlScz95rMM7'
	#access_token = '1121237479-li2tQfxLq8eCpgJu5JAG33U0ViLXgZQg2KBLhEm' 
	access_token = '2521477255-6FWcso042sTIZ3eSzHofYLnNCHsPRanJmidOzF0'
	#access_token_secret = 'm9Ucvzl6HkqE1yxrK946BsEu6wfMBDI9YoXlnUbpz95Af' 
	access_token_secret = 'wWAlsb14aYwp3LwQOimyPdwam5aE3B2x1h4XEyYyvuuy4'
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	all_tweets = []

	# get today's tweets
	# tweets = api.search(query, lang='en', rpp=100, count=100, result_type='recent')
	# for tweet in tweets:
	# 	all_tweets.append(tweet)
	# 	#print tweet.created_at
	# 	tweet.created_at = tweet.created_at.date()
	# 	#print tweet.text
	# get tweets for past week
	for date in last_dates(7):
		print "UNTIL THIS DATE: " + str(date)
		tweets = api.search(query, lang='en', rpp=100, count=100, result_type='mixed', until=str(date))
		for tweet in tweets:
			all_tweets.append(tweet)
			print tweet.created_at
			tweet.created_at = tweet.created_at.date()
			#print tweet.text
	#print "HERE!!!!!!"
	#print len(all_tweets)
	return all_tweets

# writes training set to file for the michigan data set.
# currently: IN USE
def write_training_set_mich(filename):
	word_freq = {}
	stopwords = []
	with open("training/stopwords.txt") as f:
		for line in f:
			stopwords.append(line.rstrip('\n').lower())

	labels = []
	texts = []
	with open(filename) as f:
		for line in f:
			temp = line.split('\t')
			labels.append('pos' if temp[0] == '1' else 'neg')
			words = temp[1].rstrip('\n').lower().split(' ')
			texts.append(words)
			for w in words:
				if w not in stopwords:
					if w in word_freq:
						word_freq[w] += 1
					else:
						word_freq[w] = 1
	features = []
	for t in texts:
		features.append([1 if w in t else 0 for w in word_freq.keys()])
	with open('training/umich/results.txt', 'w') as fout:
		pickle.dump(features, fout)
	with open('training/umich/labels.txt', 'w') as fout:
		pickle.dump(labels, fout)
	with open('training/umich/wordslist.txt', 'w') as fout:
		pickle.dump(word_freq.keys(), fout)
	return

# writes training set for the movie review training set. currently: NOT IN USE
#numwords is the number of keys to consider. most common numwords number of keys will be used as feature vectors
#numsamples number of reviews to use (total 5331)
def write_training_set(numwords=500, numsamples=1000):
	word_freq = {}
	stopwords = []
	with open("training/stopwords.txt") as f:
		for line in f:
			stopwords.append(line.rstrip('\n').lower())
	#words_list = set()

	positive = []
	i = 0
	with open('training/rt-polaritydata/rt-polarity.pos') as f:
		for line in f:
			i += 1
			if i <= numsamples:
				words = line.rstrip('\n').lower().split()
				positive.append(words)
				#words_list |= set(words)
				for w in words:
					if w not in stopwords:
						if w in word_freq:
							word_freq[w] += 1
						else:
							word_freq[w] = 1

	negative = []
	i = 0
	with open('training/rt-polaritydata/rt-polarity.neg') as f:
		for line in f:
			i += 1
			if i <= numsamples:
				words = line.rstrip('\n').lower().split()
				negative.append(words)
				#words_list |= set(words)
				for w in words:
					if w not in stopwords:
						if w in word_freq:
							word_freq[w] += 1
						else:
							word_freq[w] = 1

	words_list_truncated = zip(*sorted(word_freq.items(), key=lambda x:x[1], reverse=True))[0][0:numwords]

	train_pos = []
	train_neg = []
	for p in positive:
		train_pos.append([1 if w in p else 0 for w in words_list_truncated])
	with open('training/rt-polaritydata/training_pos.txt', 'w') as fout:
		pickle.dump(train_pos, fout)
	for n in negative:
		train_neg.append([1 if w in n else 0 for w in words_list_truncated])
	with open('training/rt-polaritydata/training_neg.txt', 'w') as fout:
		pickle.dump(train_neg, fout)
	with open('training/rt-polaritydata/words_list_truncated.txt', 'w') as fout:
		pickle.dump(words_list_truncated, fout)
	#print "done"

def load_data(filename):
	with open(filename) as f:
		data = pickle.load(f)
	return data

# def get_sentiment_data(query, training_set):
# 	tweets = grab_tweets(query)
# 	#print "****HERE****"
# 	#train_pos = load_data('training_pos.txt')
# 	#train_neg = load_data('training_neg.txt')
# 	#words_list = load_data('words_list_truncated.txt')
# 	#clf.fit(train_pos+train_neg, ['pos']*len(train_pos) + ['neg']*len(train_neg))

# 	train = load_data('training/' + training_set + '/results.txt')
# 	#print "HERE"
# 	labels = load_data('training/' + training_set + '/labels.txt')
# 	words_list = load_data('training/' + training_set + '/wordslist.txt')

# 	#print labels 
# 	clf = BernoulliNB(binarize=None)
# 	clf.fit(train, labels)
	# classified = {}
	# for tweet in tweets:
	# 	if tweet.created_at in classified.keys():
	# 		classified[tweet.created_at] = classified[tweet.created_at] + [classify(tweet.text, clf, words_list)[0]]
	# 	else:
	# 		classified[tweet.created_at] = [classify(tweet.text, clf, words_list)[0]]
	# #print classified

	# returndata = {}
	# for key in classified:
	# 	numpos = sum([1 if v=='pos' else 0 for v in classified[key]])
	# 	#returndata[key] = (numpos, len(classified[key]) - numpos) #tuple of positive, negative
	# 	# percent:
	# 	#returndata[key] = float(sum([1 if v == 'pos' else 0 for v in classified[key]]))/len(classified[key])
	# 	returndata[key] = ceil(float(sum([1 if v == 'pos' else 0 for v in classified[key]]))/len(classified[key])*100)/100.0
	# return returndata
	# #print percents

def get_features(original):
	stopwords = []
	with open("training/stopwords.txt") as f:
		for line in f:
			stopwords.append(line.rstrip('\n').lower())

	temp = original.rstrip('\n').rstrip('.').rstrip('!').rstrip('?')
	tempset = set(temp.lower().split())
	tempset = tempset.difference(stopwords)
	features = {}
	for t in tempset:
		features[t] = True
	return features

def get_sentiment_data(query, training_set):
	train = []
	with open('training/' + training_set + '/training.txt') as f:
		for line in f:
			temp = line.split('\t')
			#print temp
			train.append((get_features(temp[1]), temp[0]))
	clf = NaiveBayesClassifier.train(train)

	tweets = grab_tweets(query)
	print "HERE"
	classified = {}
	for tweet in tweets:
		if tweet.created_at in classified.keys():
			classified[tweet.created_at] = classified[tweet.created_at] + [clf.classify(get_features(tweet.text))]
		else:
			classified[tweet.created_at] = [clf.classify(get_features(tweet.text))]
	print classified

	returndata = {}
	for key in classified:
		#numpos = sum([1 if v=='pos' else 0 for v in classified[key]])
		#returndata[key] = (numpos, len(classified[key]) - numpos) #tuple of positive, negative
		# percent:
		returndata[key] = float(sum([1 if v == '1' else 0 for v in classified[key]]))/len(classified[key])
		#returndata[key] = math.ceil(float(sum([1 if v == '1' else 0 for v in classified[key]]))/len(classified[key])*100)/100.0
	print returndata
	return returndata

# def classify(query, clf, words_list):
# 	tokens = query.rstrip('\n').lower().split()
# 	feature = [1 if w in tokens else 0 for w in words_list]
# 	return clf.predict(feature)

if __name__ == "__main__":
	#write_training_set_mich('training/umich/training.txt')
	data = get_sentiment_data('ucsb', 'sentiment140')
	dates_list, percents_list = zip(*sorted(data.items(), key=lambda x:x[0]))
	print dates_list
	print percents_list
