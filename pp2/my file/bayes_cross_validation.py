
# comp136 project2 Task 3
# beibei du
#  10/27/2016
import datetime
import os
import sys
import math
import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt

# to store optimal lambda
lam_list = []

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
			result.append(float(line))
	return result

# calculate w
def calculate_w(lam, tr, tr_r):
	feature = len(tr[0])
	example = len(tr)
	iden = np.identity(feature)
	iden = np.dot(lam, iden)
	tr_trans = np.transpose(tr)
	sum_1 = np.add(np.dot(tr_trans,tr),iden)
	sum_inverse = inv(sum_1)
	prod = np.dot(sum_inverse, tr_trans)
	w = np.dot(prod,tr_r)
	return w

# calculate mse
def calculate_mse(w, t, t_r):
	N = len(t)
	pro = np.dot(t, w)
	dif = np.subtract(pro, t_r)
	squ = np.square(dif)
	mse = np.mean(squ)
	return mse

# cross validation to pick lambda
def cross_val(tr, tr_r):
	tr_size = len(tr)
	interval = tr_size / 10
	lam_mse = []  # record mse_avg for every lam
	for lam in range(151):
		mse_list = [] # record mse for every splits
		for i in range(10):
			start_index = i * interval
			if i == 9:
				end_index = tr_size
			else:
				end_index = (i+1) * interval
			tr_1 = tr[0:start_index]
			tr_r_1 = tr_r[0:start_index]
			train = []
			train_r = []
			train_test = tr[start_index:end_index]
			train_test_r = tr_r[start_index:end_index]
			tr_2 = tr[end_index:]
			tr_r_2 = tr_r[end_index:]
			train = np.concatenate((tr_1, tr_2))
			train_r = np.array(tr_r_1 + tr_r_2)
			w = calculate_w(lam, train, train_r)
			mse = calculate_mse(w, train_test, train_test_r)
			mse_list.append(mse)
		mse_avg = np.mean(mse_list)
		lam_mse.append(mse_avg)

	# return lam with minimum mse value
	lam_min = lam_mse.index(min(lam_mse))
	return lam_min

def Regularization(tr, tr_r, te, te_r):
	global lam_list
	lam = cross_val(tr, tr_r)
	lam_list.append(lam)
	w = calculate_w(lam, tr, tr_r)
	mse_te = calculate_mse(w, te, te_r)
	return mse_te

def main():
	global lam_list
	tr_file = ['train-crime.csv', 'train-wine.csv','train-100-10.csv', 'train-100-100.csv', 'train-1000-100.csv']
	tr_r_file = ['trainR-crime.csv', 'trainR-wine.csv','trainR-100-10.csv', 'trainR-100-100.csv',  'trainR-1000-100.csv']
	te_file = ['test-crime.csv', 'test-wine.csv','test-100-10.csv', 'test-100-100.csv', 'test-1000-100.csv']
	te_r_file = ['testR-crime.csv', 'testR-wine.csv','testR-100-10.csv', 'testR-100-100.csv',  'testR-1000-100.csv']
	
	te_mse = []
	file_number = len(tr_file)
	run_time = []
	for i in range(file_number):
	# start time
		a = datetime.datetime.now()
		tr = read_file(tr_file[i])
		tr_r = read_file_r(tr_r_file[i])
		te = read_file(te_file[i])
		te_r = read_file_r(te_r_file[i])
		if tr_file[i] == 'train-1000-100.csv':
			MSE_te = Regularization(tr, tr_r, te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			MSE_te = Regularization(tr[0:50], tr_r[0:50], te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			MSE_te = Regularization(tr[0:100], tr_r[0:100], te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			MSE_te = Regularization(tr[0:150], tr_r[0:150], te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
			# Task 1 Regularization
		else:
			MSE_te = Regularization(tr, tr_r, te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
	
	#print "run time:"
	#print run_time
	print "lam list"
	print lam_list
	print "test mse is:"
	print te_mse


	# run time: [1.850564, 0.214724, 0.128529, 1.296833, 3.708794, 4.776038, 6.029359, 7.403829]
	#lam list :[150, 2, 12, 20, 39, 24, 30, 46]
	#test mse is:[0.39233899203438133, 0.62530884230477923, 4.1757091596711433, 5.0808888179185328, 4.3227223508824659, 5.9344653841860655, 5.2599827410154809, 4.9341926077600835]
main()



		

