import numpy as np 
import random
from numpy.linalg import inv
from random import shuffle
#import matplotlib.pyplot as plt 
from collections import OrderedDict
import csv


def read_file(filename):
	with open(filename, 'r') as f:
		content = f.read().split()
	return content
	

def main():
	filename = range(1, 11)
	K = 2
	N_iters = 500
	data = []
	doc_len = []
	
	# array of document indices d_n
	d_n = []
	# array of initial topic indices z_n
	z_n = []
	for i in filename:
		res = read_file(str(i))
		doc_len.append(len(res))
		temp = [str(i)] * len(res)
		d_n += temp
		data.append(res)
	# array of words indices w(n)
	words = [w for d in data for w in d]
	vocab = list(set(words))
	word_indices = OrderedDict()
	for v in vocab:
		word_indices[v] = vocab.index(v)
	print word_indices

	w_n = []
	for word in words:
		w_n.append(word_indices[word])

	# total number of N_words
	N_words = len(w_n)
	# array of initial topic indices
	for i in range(N_words):
		z_n.append(str(random.choice(range(1, K+1))))
	
	# get vocabulary
	"""
	vocab = dict()
	for w in w_n:
		if w in vocab.keys():
			vocab[w] += 1
		else:
			vocab[w] = 1
	# number of words in the vocabulary
	V = len(vocab.keys())
	"""
	
	# dict to ordered dict
	#o_vocab = OrderedDict(sorted(vocab.items(), key=lambda x: x[0]))
	
	V = len(vocab)
	alpha = float(50) / float(K)
	alpha_1 = alpha * np.ones(K)
	beta = 0.1
	beta_1 = beta * np.ones(V)
	
	# random permutation of N_words
	pi_n = np.random.permutation(range(0, N_words))
	
	# initialize a D * K matrix C_d
	D = len(filename)
	C_d = []
	i = 0
	k = 0
	while i < N_words:
		j = i
		temp = [0] * K
		end = j + doc_len[k]
		while j < end:
			temp[int(z_n[j]) - 1] += 1
			j += 1
		C_d.append(temp)
		i = j
		k += 1
	
	# initialize a K * V matrix C_t
	C_t = []
	for i in range(K):
		temp = [0] * V
		C_t.append(temp)

	#v_key = o_vocab.keys()
	for i in range(N_words):
		topic = z_n[i]
		topic_index = int(topic) - 1
		#word = w_n[i]
		#word_index = v_key.index(word)
		word_index = w_n[i]
		C_t[topic_index][word_index] += 1

	# initialize a 1 * K array of probabilities P (to zero)
	P = [0] * K
	#print P
	# step 5
	for i in range(N_iters):
	#for i in range(2):
		for n in range(N_words):
			index = pi_n[n]
			word = w_n[index]
			topic = z_n[index]
			doc = d_n[index]
			C_d[int(doc) - 1][int(topic) - 1] -= 1
			#C_t[int(topic) - 1][v_key.index(word)] -= 1
			C_t[int(topic) - 1][word] -= 1
			for k in range(K):
				#p_1 = (C_t[k][v_key.index(word)] + beta) / (V * beta + sum(C_t[k]))
				p_1 = (C_t[k][word] + beta) / (V * beta + sum(C_t[k]))

				p_2 = (C_d[int(doc)-1][k] + alpha) / (K * alpha + sum(C_d[int(doc) - 1]))
				P[k] = p_1 * p_2
			# normalize P
			total = sum(P)
			for k in range(K):
				P[k] /= float(total)
			
			#print C_t
			r = random.uniform(0,1)
			for k in range(K):
				if r >= sum(P[:k]) and r <= sum(P[:k+1]):
					topic = str(k+1) 
					break
			z_n[index] = topic
			C_d[int(doc) - 1][int(topic) - 1] += 1
			#C_t[int(topic) - 1][v_key.index(word)] += 1
			C_t[int(topic) - 1][word] += 1
			
			

			"""
	print "z_n"
	print z_n
	print "C_d and C_t"
	print C_d
	print C_t
	"""
	#print C_t
	fre_word = []
	for k in range(K):
		fre = []
		temp = sorted(C_t[k], reverse = True)
		for t in range(3):
			index = C_t[k].index(temp[t])
			for key, val in word_indices.iteritems():
				if val == index:
					fre.append(key)
		fre_word.append(fre)
	print fre_word
	file = open("topicwords.csv", "w")
	wr = csv.writer(file, dialect = 'excel')
	wr.writerows(fre_word)
	file.close()



			
		

				




		










main()