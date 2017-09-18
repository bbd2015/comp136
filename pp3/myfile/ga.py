import os
import sys
import math
import numpy as np 
import random
from numpy.linalg import inv
from random import shuffle
import datetime
import matplotlib.pyplot as plt

alpha = 0.1
eta = 10**(-3)

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
	return 1 / (1 + np.exp(-a))

# compute R
def compute_R(y):
	r = map(lambda x: x * (1 - x), y)
	return np.diagflat(r)

# compute w
def compute_w(tr, tr_r, w0):
	global alpha
	global eta
	t = np.array(tr_r).astype(float)
	y = compute_y(w0, np.transpose(tr))
	tr_transpose = np.transpose(tr)
	p1 = np.dot(tr_transpose, (y - t)) + np.dot(alpha, w0)
	w1 = w0 - np.dot(eta, p1)
	return w1

# compute sN
def compute_sN(w, tr):
	y = compute_y(w, np.transpose(tr))
	R = compute_R(y)
	l = len(y)
	s = 0
	global alpha
	for i in range(l):
		p1 = np.dot(R[i][i], tr[i])
		p2 = p1[:,None]*tr[i]
		s += p2
	s0 = inv(np.identity(len(tr[0])) / alpha)
	sN = inv(s0 + s)
	return sN

# compute error
def compute_error(w, tr, te_N, data, data_label):
	sN = compute_sN(w, tr)
	error = 0
	for i in range(te_N):
		ua = np.dot(w, data[i])
		sig_s = np.dot(np.dot(data[i], sN), data[i])
		a = ua / (math.sqrt(1 + np.pi * sig_s / 8))
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
	global eta
	global alpha
	file_time = []
	file_error =[]
	w_file = []
	for i in range(file_number):
		data = read_file(data_file[i])
		data_label = read_file_r(label_file[i])
		N = len(data)
		d = len(data[0])
		temp = np.ones((N, d+1))
		temp[:,:-1] = data
		data = temp
		te_N = int(N/3)
		tr = data[te_N:]
		tr_r = data_label[te_N:]
		error_list = []
		# run 3 times
		time_list = []
		w_list =[]
		for t in range(3):
			run_time = []
			w0 = [0] * len(tr[0])
			# clock time start
			a = datetime.datetime.now()
			w = compute_w(tr, tr_r, w0)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			if t == 0:
				w_list.append(w)
				error_rate = compute_error(w, tr, te_N, data, data_label)
				error_list.append(error_rate)
			sum1 = 1
			n = 1
			while n < 6000 and sum1 >= eta: 
				w0 = w
				a = datetime.datetime.now()
				w = compute_w(tr, tr_r, w0)
				b = datetime.datetime.now()
				run_time.append(run_time[-1] + (b-a).total_seconds())
				if t == 0:
					w_list.append(w)
					error_rate = compute_error(w, tr, te_N, data, data_label)
					error_list.append(error_rate)
				sum1 = sum(np.square(np.subtract(w, w0))) / sum(np.square(w0))
				n += 1
			time_list.append(run_time)
		w_file.append(w_list)
		time_list = np.mean(time_list, axis = 0)
		file_time.append(time_list)		
		file_error.append(error_list)


	"""
	print "Gradient:"
	for k in range(2):
		print data_file[k]
		print file_time[k]
		print file_error[k]
		#print w_file[k]

	a_newton_time = [ 0.01537367,  0.02855167,  0.04108567,  0.05452367,  0.068228  ]
	a_newton_error = [0.05855855855855856, 0.04804804804804805, 0.046546546546546545, 0.046546546546546545, 0.046546546546546545]
	plt.figure(1)
	x = file_time[0]
	y = file_error[0]
	plt.plot(x, y)
	plt.plot(a_newton_time, a_newton_error)
	plt.xlabel('time line')
	plt.ylabel('error rate')
	plt.title('A data')
	plt.gcf().autofmt_xdate()
	plt.savefig('task2_A.png')
	plt.clf()
	

	plt.figure(2)
	usps_newton_time = [ 0.02759267,  0.05277067 , 0.078221 ,   0.10310333,  0.12794833 , 0.15363867,
  						0.17856  ,   0.204489 ,   0.22854267]
  	usps_newton_error = [0.04093567251461988, 0.042884990253411304, 0.03313840155945419, 0.037037037037037035, 0.03898635477582846, 0.037037037037037035, 0.03313840155945419, 0.03313840155945419, 0.03508771929824561]
	x = file_time[1]
	y = file_error[1]
	plt.plot(x, y)
	plt.plot(usps_newton_time, usps_newton_error)
	plt.xlabel('time line')
	plt.ylabel('error rate')
	plt.title('USPS data')
	plt.gcf().autofmt_xdate()
	plt.savefig('task2_usps.png')
	plt.clf()
	"""
main()

























