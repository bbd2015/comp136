import os
import sys
import math
import numpy as np 
import random
from numpy.linalg import inv
from collections import OrderedDict
from random import shuffle
#import matplotlib.pyplot as plt

alpha = 0.1

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
############################################################
# generative





############################################################
# discrimitive 
# compute y
def compute_y (w0, phi):
	a = np.dot(w0, phi)
	return 1 / (1 + np.exp(-a))

# compute R
def compute_R(y):
	r = map(lambda x: x * (1 - x), y)
	return np.diagflat(r)

# compute w
def compute_w(phi, phi_tranpose, R, y, t, w0):
	global alpha
	N = len(phi[0])
	p1 = inv(np.dot(alpha, np.identity(N)) + np.dot(np.dot(phi_tranpose, R), phi))
	p2 = np.dot(phi_tranpose, (y-t)) + np.dot(alpha, w0)
	w1 = w0 - np.dot(p1, p2)
	return w1

def compute_w_sN(data, data_label, tr_index):
	global alpha
	phi = []
	t = []
	for i in tr_index:
		phi.append(data[i])
		t.append(data_label[i])

	t = np.array(t).astype(float)
	N = len(phi[0])
	w0 = [0] * N
	phi_tranpose = np.transpose(phi)
	a = np.dot(w0, phi_tranpose)
	y = 1 / (1 + np.exp(-a))
	
	R = map(lambda x: x * (1 - x), y)
	R = np.diagflat(R)
	
	# compute w1
	w1 = compute_w(phi, phi_tranpose, R, y, t, w0)
	#sum1 = sum(np.square(np.subtract(w1, w0))) / sum(np.square(w0))
	sum1 = 1
	n = 1
	while n < 100 and sum1 >= 10 ** (-3):
		w0 = w1
		y = compute_y(w0, phi_tranpose)
		R = compute_R(y)
		w1 = compute_w(phi, phi_tranpose, R, y, t, w0)
		sum1 = sum(np.square(np.subtract(w1, w0))) / sum(np.square(w0))
		n += 1

	y = compute_y(w1, phi_tranpose)
	R = compute_R(y)
	l = len(y)
	s = 0
	for i in range(l):
		p1 = np.dot(R[i][i], phi[i])
		p2 = p1[:,None]*phi[i]
		s += p2
	s0 = inv(np.identity(N) / alpha)
	sN = inv(s0 + s)
	return w1, sN

# compute error
def compute_error(data, data_label, index_test, w_map, sN):
	error = 0
	for te_i in index_test:
		ua = np.dot(w_map, data[te_i])
		sig_s = np.dot(np.dot(data[te_i], sN), data[te_i])
		a = ua / math.sqrt(1 + np.pi * sig_s / 8)
		if a >= 0:
			if data_label[te_i] != '1':
				error += 1
		else:
			if data_label[te_i] != '0':
				error += 1
	return error / float(len(index_test))

#############################################################################

def main():
	data_file = ['A.csv', 'B.csv', 'usps.csv']
	label_file = ['labels-A.csv', 'labels-B.csv', 'labels-usps.csv']
	file_number = len(data_file)

	# to store error list of different data set
	ge_error_file = []
	dis_error_file = []
	for i in range(file_number):
		dis_error_list = []
		data = read_file(data_file[i])
		data_label = read_file_r(label_file[i])

		index_N = range(N)

		# run 30 times
		for t in range(30):
			# set up 1/3 data set index
			te_N = int(N/3)
			index_test = np.random.choice(N, te_N, replace = False)
			# remain 2/3 train set
			index_train = [v for j, v in enumerate(index_N) if j not in index_test]

			# Record performance for discrimitive method
			#splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
			tr_N = len(index_train)
			splits = np.arange(0.05, 1.05, 0.05)

			dis_error_rate = []
			for s in splits:
				tr_index = np.random.choice(index_train, int(s), replace = False)
				w_map, sN = compute_w_sN(data, data_label, tr_index)
				# test file
				error = compute_error(data, data_label, index_test, w_map, sN)
				dis_error_rate.append(error)

			dis_error_list.append(dis_error_rate)
		dis_error_file.append(dis_error_list)
			
	print "Dicriminative:"
	for k in range(3):
		A_mean = np.mean(dis_error_file[k], axis = 0)
		A_std = np.std(dis_error_file[k], axis = 0)
		print data_file[k]
		print A_mean
		print A_std
		
main()

























