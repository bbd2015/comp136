# beibei du
import numpy as np
import math 
#import matplotlib.pyplot as plt
from unigram_model import *


def main():
	# read train data set and test data set into list, also build vocab list 
	# from train data and test data which include distinct words
	train_set = open ('training_data.txt', 'r').read().split()
	test_set = open ('test_data.txt', 'r').read().split()
	vocab_set = []

	# build vocab from the entire test and train sets
	for data in train_set:
		if data not in vocab_set:
			vocab_set.append(data)

	for data in test_set:
		if data not in vocab_set:
			vocab_set.append(data)

	N = len(train_set)
	K = len(vocab_set)

	################################################
	# Task 1: Model Training, Prediction, Evaluation

	train_splits = [128.0, 64.0, 16.0, 4.0, 1.0]
	# perplexity list of test data using maximum likelihood estimate
	mle_perpelxity = []
	# perplexity list of test data using MAP estimate
	map_perpelxity = []
	# perplexity list of test data using predictive distribution
	pre_perplexity = []
	# train set size
	Train = []
	i = 0
	while i < len(train_splits):
		train_size = int (round(N / train_splits[i]))
		Train.append(train_size)

		max_like_estimate = MLE_Model(vocab_set)
		max_like_estimate.train(train_set[:train_size])
		mle_perpelxity.append(max_like_estimate.perplexity(test_set))
 
		alpha = 2 * np.ones(K)
		map_estimate = MAP_Model(vocab_set,alpha)
		map_estimate.train(train_set[:train_size])
		map_perpelxity.append(map_estimate.perplexity(test_set))

		pred_estimate = PRED_Model(vocab_set, alpha)
		pred_estimate.train(train_set[:train_size])
		pre_perplexity.append(pred_estimate.perplexity(test_set))

		i += 1

	print "mle perpelxity: "
	print mle_perpelxity
	print "map perplexity:"
	print map_perpelxity
	print "predictive perplexity"
	print pre_perplexity

	################################################
	# Task 2: Model Selection
	alpha_prime = range(1,11)
	train_size = int(round(N / 128))
	log_evidence = []
	pre_perplexity_2 = []

	for alpha in alpha_prime:
		alphas = alpha * np.ones(K)
		pred_estimate = PRED_Model(vocab_set, alphas)
		pred_estimate.train(test_set[:train_size])
		log_evidence.append(pred_estimate.evidence())
		pre_perplexity_2.append(pred_estimate.perplexity(test_set))

	print "predictive perplexity 2:"
	print pre_perplexity_2
	print "log evidence: "
	print log_evidence

main()


















