# comp136 project2 Task 4
# beibei du
# 10/27/2016
import os
import sys
import math
import numpy as np 
from numpy.linalg import inv
from numpy import linalg as la
import datetime

counter_list = []
# store run time
run_time = []
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

def select_mN (tr, tr_r, alpha, beta):
	# get eigenvalue 
	tr_trans = np.transpose(tr)
	pro = np.dot(tr_trans, tr)
	beta_phi_pro = np.dot(beta, pro)
	lamdas = la.eigvals(beta_phi_pro)
	

	# get transpose of sn <3.54>
	size = len(beta_phi_pro)
	iden = np.identity(size)
	sN_inverse = np.add(np.dot(alpha, iden), beta_phi_pro)
	sN =  inv(sN_inverse)

	# get Mn <3.53>
	phi_t =  np.dot(tr_trans, tr_r)
	sN_phi_t = np.dot(sN, phi_t)
	mN = np.dot(beta, sN_phi_t)

	# get gamma <3.91>
	gamma = 0.0
	for lam in lamdas:
		if isinstance(lam, complex):
			index = np.where(lamdas == lam)
			lam = lam.real
			lamdas[index] = lam
		gamma += (float(lam) / float(lam + alpha))

	# get alpha <3.92>
	mN_trans = np.transpose(mN)
	mN_pro = np.dot(mN_trans, mN)
	alpha_new = gamma / mN_pro

	# get beta (3.95>
	N = len(tr)
	mN_phi = np.dot(tr, mN)
	diff = np.subtract(tr_r, mN_phi)
	sum_squ = np.sum(np.square(diff))
	beta_new = sum_squ / float(N - gamma) 
	beta_new = np.reciprocal(beta_new)

	return mN, alpha_new, beta_new


# modle select to pick mN (w)
def model_select(tr, tr_r):
	global counter_list
	alpha = 1.0
	beta = 1.0
	mN, alpha_new, beta_new= select_mN(tr, tr_r, alpha, beta)
	count = 0
	while (abs(alpha - alpha_new) >= 0.00001 ) and (abs(beta - beta_new) >= 0.00001):
		alpha, beta = alpha_new, beta_new
		mN, alpha_new, beta_new= select_mN(tr, tr_r, alpha, beta)
		count = count + 1
	else:
		counter_list.append(count)
		return mN
	

def Regularization(tr, tr_r, te, te_r):
	w = model_select(tr, tr_r)
	mse_te = calculate_mse(w, te, te_r)
	return mse_te

def main():
	global counter_list
	global run_time
	tr_file = ['train-crime.csv', 'train-wine.csv','train-100-10.csv', 'train-100-100.csv','train-1000-100.csv']
	tr_r_file = ['trainR-crime.csv', 'trainR-wine.csv', 'trainR-100-10.csv', 'trainR-100-100.csv','trainR-1000-100.csv']
	te_file = ['test-crime.csv', 'test-wine.csv','test-100-10.csv','test-100-100.csv', 'test-1000-100.csv']
	te_r_file = ['testR-crime.csv', 'testR-wine.csv', 'testR-100-10.csv', 'testR-100-100.csv', 'testR-1000-100.csv']
	

	te_mse = []

	file_number = len(tr_file)
	for i in range(file_number):
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
	
		else:
			MSE_te = Regularization(tr, tr_r, te, te_r)
			te_mse.append(MSE_te)
			b = datetime.datetime.now()
			run_time.append((b-a).total_seconds())
	#print tr_file
	#print "counter:"
	#print counter_list
	#print "run time:"
	#print run_time
	print "test mse is:"
	print te_mse
# run time: [0.151428, 0.034536, 0.007779, 0.105999, 0.117914, 0.154577, 0.183432, 0.19713]
# [11, 13, 3, 13, 3, 8, 6, 2]
#[0.39110220806201251, 0.6267448249308214, 4.1801330312023159, 7.3525369250732879, 4.3383499509291168, 5.7895752937487694, 5.7339307345248693, 5.2489966154820733]
main()



		

