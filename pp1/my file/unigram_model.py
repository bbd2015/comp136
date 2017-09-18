# Beibei Du
import math
import numpy as np


# build unigram model
class Model():
	def __init__(self, vocab):
		self.vocab = vocab
		self.K = len(vocab)

	def train(self, train_set):
		pass

	def probability(self, w):
		pass

	def perplexity(self, test_set):
		N = len(test_set)
		sum = 0.0
		for w in test_set:
			#handle ln(0) = inf case
			if self.probability(w) == 0:
				sum = sum + np.log(1e-7)
			else:
				sum = sum + np.log(self.probability(w)) 
		return (np.exp(-1.0 * sum / float(N)))

class MLE_Model(Model):
	def __init__(self, vocab):
		Model.__init__(self,vocab)

	def train(self, train_set):
		self.N = len(train_set)
		self.m = {}
		
		for word in self.vocab:
			self.m[word] = 0
		for word in train_set:
			self.m[word] += 1
		
		return self

	def probability(self, w):
		return self.m[w] / float(self.N)
	

class MAP_Model(Model):
	def __init__(self, vocab, alpha):
		Model.__init__(self,vocab)
		self.alpha = alpha
		self.alpha_zero = sum(self.alpha)
		self.alpha_map = {}
		for i in range(self.K):
			self.alpha_map[self.vocab[i]] = alpha[i]

	def train(self, train_set):
		self.N = len(train_set)
		self.m = {}
		for word in self.vocab:
			self.m[word] = 0
		for word in train_set:
			self.m[word] += 1

		return self

	def probability(self, w):
		m_k = self.m[w]
		alpha_k = self.alpha_map[w]
		N = self.N

		return float(m_k + alpha_k - 1) / float(N + self.alpha_zero - len(self.vocab))

class PRED_Model(Model):
	def __init__(self, vocab, alpha):
		Model.__init__(self, vocab)
		self.alpha = alpha
		self.alpha_zero = sum(self.alpha)
		self.alpha_map = {}
		for i in range(self.K):
			self.alpha_map[self.vocab[i]] = alpha[i]


	def train(self, train_set):
		self.N = len(train_set)
		self.m = {}
		for word in self.vocab:
			self.m[word] = 0
		for word in train_set:
			self.m[word] += 1

		return self

	def probability (self, w):
		m_k = self.m[w]
		alpha_k = self.alpha_map[w]
		N = self.N

		return float (m_k + alpha_k) / float(N + self.alpha_zero)

	def evidence(self):
		res = 0.0
		k = 1
		while k < self.alpha_zero:
			res = res + np.log(k)
			k = k + 1

		for word in self.vocab:
			m_k = self.m[word]
			alpha_k = self.alpha_map[word]
			res = res + np.log(math.gamma(m_k + alpha_k))
			res = res - np.log(math.gamma(alpha_k))

		j = 1
		while j < self.alpha_zero + self.N:
			res = res - np.log(j)
			j = j + 1
			
		return res

