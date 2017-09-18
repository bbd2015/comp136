import os
import sys
import math
import numpy as np 
import random
from numpy.linalg import inv
from collections import OrderedDict
from random import shuffle
import datetime
import matplotlib.pyplot as plt

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

# compute y
def compute_y (w0, phi):
	a = np.dot(w0, phi)
	return 1.0 / (1 + np.exp(-a))

# compute R
def compute_R(y):
	r = map(lambda x: x * (1 - x), y)
	return np.diagflat(r)

# compute w
def compute_w(tr, tr_r, w0):
	global alpha
	t = np.array(tr_r).astype(float)
	y = compute_y(w0, np.transpose(tr))
	R = compute_R(y)
	tr_transpose = np.transpose(tr)
	p1 = inv(np.dot(alpha, np.identity(len(tr[0]))) + np.dot(np.dot(tr_transpose, R), tr))
	p2 = np.dot(tr_transpose, (y-t)) + np.dot(alpha, w0)
	w1 = w0 - np.dot(p1, p2)

	return w1

# compute sN
def compute_sN(w, tr):
	y = compute_y(w, np.transpose(tr))
	R = compute_R(y)
	l = len(y)
	s = 0
	global alpha
	s = np.dot(np.dot(np.transpose(tr), R), tr)
	s0 = inv(np.identity(len(tr[0])) / alpha)
	sN = inv(s0 + s)
	return sN

# compute error
def compute_error(w, tr, te_N, data, data_label):
	# compute sN
	sN = compute_sN(w, tr)
	
	error = 0
	for i in range(te_N):
		ua = np.dot(w, data[i])
		sig_s = np.dot(np.dot(data[i], sN), data[i])
		a = ua / float(math.sqrt(1 + np.pi * sig_s / 8))
		if a >= 0:
			if data_label[i] == '0':
				error += 1
		else:
			if data_label[i] == '1':
				error += 1

	return error /float(te_N)


def main():
	data_file = ['A.csv', 'usps.csv']
	label_file = ['labels-A.csv', 'labels-usps.csv']

	file_number = len(data_file)
	global alpha
	file_time = []
	file_error =[]
	w_file = []
	for i in range(file_number):
		error_list = []
		data = read_file(data_file[i])
		N = len(data)
		d = len(data[0])
		print N
		print d
		temp = np.ones((N, d+1))
		temp[:,:-1] = data
		data = temp
		data_label = read_file_r(label_file[i])
		N = len(data)
		te_N = int(N/3)
		tr = data[te_N:]
		tr_r = data_label[te_N:]
		time_list = []
		w_list = []
		# run 3 times
		for t in range(3):
			run_time = []
			w0 = [0] * len(tr[0])
			# clock time start
			a = datetime.datetime.now()
			w = compute_w(tr, tr_r, w0)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			if t == 2:
				w_list.append(w)
				error_rate = compute_error(w, tr, te_N, data, data_label)
				error_list.append(error_rate)

			sum1 = 1
			n = 1
			while n <= 100 and sum1 >= 10 ** (-3): 
				w0 = w
				a = datetime.datetime.now()
				w = compute_w(tr, tr_r, w0)
				b = datetime.datetime.now()
				run_time.append(run_time[-1] + (b-a).total_seconds())
				if t == 2:
					w_list.append(w)
					error_rate = compute_error(w, tr, te_N, data, data_label)
					error_list.append(error_rate)
				sum1 = sum(np.square(np.subtract(w, w0))) / sum(np.square(w0))
				n += 1
			time_list.append(run_time)
		#print w_list[-1]
		w_file.append(w_list)
		time_list = np.mean(time_list, axis = 0)
		file_time.append(time_list)		
		file_error.append(error_list)
	
			
	print "Newton"
	for k in range(2):
		print data_file[k]
		print file_time[k]
		print file_error[k]
		print w_file[k]
	
	plt.figure(1)
	x = file_time[0]
	y = file_error[0]
	plt.plot(x, y)
	plt.xlabel('time line')
	plt.ylabel('error rate')
	plt.title('Newton A data')
	plt.gcf().autofmt_xdate()
	plt.savefig('newton_A.png')
	plt.clf()

	plt.figure(2)
	x = file_time[1]
	y = file_error[1]
	plt.plot(x, y)
	plt.xlabel('time line')
	plt.ylabel('error rate')
	plt.title('Newton USPS data')
	plt.gcf().autofmt_xdate()
	plt.savefig('newton_USPS.png')
	plt.clf()
		
		
main()

























