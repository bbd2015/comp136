import os
import sys
import math
import numpy as np 
import random
from numpy.linalg import inv
from random import shuffle
import matplotlib.pyplot as plt

def read_file(filename):
	result = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip('\n')
			line = line.split(',')
			result.append(line)
	result = np.array(result)
	result = result.astype(float)
	return result

def read_file_r(filename):
	result = []
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip('\n')
			result.append(line)
	return result

# compute u1(class 1), u2(class 0)
def compute_mean(data_file, label_file, tr_index):
	N1 = 0
	N2 = 0
	sum1 = 0
	sum2 = 0
	for i in tr_index:
		if label_file[i] == '1':
			sum1 += data_file[i]
			N1 += 1
		else:
			sum2 += data_file[i]
			N2 += 1
	u1 = sum1 / float(N1)
	u2 = sum2 / float(N2)
	return u1, u2, N1, N2

# compute s1 (class 1), s2(class 0)
def compute_s(data, data_label, tr_index, u1, u2):
	s1 = 0
	s2 = 0
	for i in tr_index:
		if data_label[i] == '1':
			dif = data[i] - u1
			s1 += dif[:,None]*dif
		else:
			dif = data[i] - u2
			s2 += dif[:,None]*dif
	return s1, s2

# compute w0 and w
def compute_w(s, u1, u2, N1, N2):
	s_inverse = inv(s)
	prod1 = np.dot(np.dot(u1, s_inverse),u1)
	prod2 = np.dot(np.dot(u2, s_inverse), u2)
	w0 = (-1) * prod1 / 2.0 + prod2 / 2.0 + math.log(float(N1)/N2)
	w = np.dot(s_inverse, u1 - u2)
	return w0, w

# compute error rate
def compute_error(data, data_label, index_test, w, w0):
	error = 0
	for te_i in index_test:
		a = np.dot(w, data[te_i]) + w0
		if a >= 0:
			if data_label[te_i] != '1':
					error += 1
		else:
			if data_label[te_i] != '0':
				error += 1
	return error / float(len(index_test))


def main():
	data_file = ['A.csv', 'B.csv', 'usps.csv']
	label_file = ['labels-A.csv', 'labels-B.csv', 'labels-usps.csv']
	#data_file = [ 'B.csv', 'usps.csv']
	#label_file = [ 'labels-B.csv', 'labels-usps.csv']

	file_number = len(data_file)
	ge_error_file = []
	for i in range(file_number):
		ge_error_list = []
		data = read_file(data_file[i])
		data_label = read_file_r(label_file[i])
		N = len(data)
		print N
		print int(N/3)
		print N - int(N/3)
		index_N = range(N)
		# run 30 times
		for t in range(30):
			
			# set up 1/3 data set index
			te_N = int(N/3)
			index_test = np.random.choice(N, te_N, replace = False)
			
			# remain 2/3 train set
			index_train = [v for j, v in enumerate(index_N) if j not in index_test]
			tr_N = len(index_train)
			# Record performance
			splits = np.arange(0.05, 1.05, 0.05)

			error_rate = []
			for s in splits:
				# train set index
				tr_index = np.random.choice(index_train, int(s * tr_N), replace = False)
				# u1: mean of class 1, u2: mean of class 0
				u1, u2, N1, N2 = compute_mean(data, data_label, tr_index)
				s1, s2 = compute_s(data, data_label, tr_index, u1, u2)
				s1 = s1 / float(N1)
				s2 = s2 / float(N2)
				# compute S
				Sig = (N1 / float(N)) * s1 +  (N2 / float(N)) * s2 
				w0, w = compute_w(Sig, u1, u2, N1, N2)

				# test file
				error = compute_error(data, data_label, index_test, w, w0)
				error_rate.append(error)
			ge_error_list.append(error_rate)
		ge_error_file.append(ge_error_list)
		

	for k in range(3):
		A_mean = np.mean(ge_error_file[k], axis = 0)
		A_std = np.std(ge_error_file[k], axis = 0)
		print data_file[k]
		print A_mean
		print A_std
 
		
main()













