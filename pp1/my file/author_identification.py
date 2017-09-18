#beibei du
from unigram_model import *
import numpy as np
import math


# Task 3: Author Identification
def main():
	data_1 = open('pg84.txt.clean.txt', 'r').read().split()
	data_2 = open('pg345.txt.clean.txt', 'r').read().split()
	data_3 = open('pg1188.txt.clean.txt', 'r').read().split()
	dic = []

	for data in data_1:
		if data not in dic:
			dic.append(data)

	for data in data_2: 
		if data not in dic:
			dic.append(data)

	for data in data_3:
		if data not in dic:
			dic.append(data)
			
	print len(dic)
	alpha = 2 * np.ones(len(dic))
	pred_estimate = PRED_Model(dic, alpha)
	pred_estimate.train(data_2)
	perplexity_1 = pred_estimate.perplexity(data_1)
	print "perplexity using pg84.txt file is %f" %perplexity_1
	perplexity_2 = pred_estimate.perplexity(data_3)
	print "perplexity using pg1188.txt file is %f"%perplexity_2
main()