# comp136 project2 Task 1
# beibei du
# 10/27/2016
import os
import sys
import math
import numpy as np 
from numpy.linalg import inv
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
	mse = np.sum(squ) / float(N)
	return mse

def Regularization(tr, tr_r, te, te_r):
	MSE_tr = []
	MSE_te = []
	for lam in range(151):
			w = calculate_w(lam, tr, tr_r)
			mse_train = calculate_mse(w, tr, tr_r)
			mse_te = calculate_mse(w, te, te_r)
			MSE_tr.append(mse_train)
			MSE_te.append(mse_te)
	return MSE_tr, MSE_te

def main():
	
	tr_file = ['train-crime.csv', 'train-wine.csv','train-100-10.csv', 'train-100-100.csv', 'train-1000-100.csv']
	tr_r_file = ['trainR-crime.csv', 'trainR-wine.csv', 'trainR-100-10.csv', 'trainR-100-100.csv', 'trainR-1000-100.csv']
	te_file = ['test-crime.csv', 'test-wine.csv','test-100-10.csv', 'test-100-100.csv', 'test-1000-100.csv']
	te_r_file = ['testR-crime.csv', 'testR-wine.csv','testR-100-10.csv', 'testR-100-100.csv',  'testR-1000-100.csv']
	

	tr_mse = []
	te_mse = []

	file_number = len(tr_file)
	for i in range(file_number):
		tr = read_file(tr_file[i])
		tr_r = read_file_r(tr_r_file[i])
		te = read_file(te_file[i])
		te_r = read_file_r(te_r_file[i])
		if tr_file[i] == 'train-1000-100.csv':
			MSE_tr, MSE_te = Regularization(tr, tr_r, te, te_r)
			tr_mse.append(MSE_tr)
			te_mse.append(MSE_te)
			
			MSE_tr, MSE_te = Regularization(tr[0:50], tr_r[0:50], te, te_r)
			tr_mse.append(MSE_tr)
			te_mse.append(MSE_te)
			MSE_tr, MSE_te = Regularization(tr[0:100], tr_r[0:100], te, te_r)
			tr_mse.append(MSE_tr)
			te_mse.append(MSE_te)
			MSE_tr, MSE_te = Regularization(tr[0:150], tr_r[0:150], te, te_r)
			tr_mse.append(MSE_tr)
			te_mse.append(MSE_te)
			
		# Task 1 Regularization
		else:
			MSE_tr, MSE_te = Regularization(tr, tr_r, te, te_r)
			tr_mse.append(MSE_tr)
			te_mse.append(MSE_te)

	#print tr_mse
	#print te_mse

	"150  opt lamda:23, opt tse: 4.84894305335" 
	"100  opt lamda:19, opt tse: 5.20591195733" 
	"50   opt lamda: 8, opt tse: 5.54090222919" 
	"1000-100 opt lamda: 27, opt tse: 4.31557063032" 
	"100-100 opt lamda: 22, opt tse: 5.07829980059" 
	"100-10 opt lamda: 8 , opt tse: 4.15967850948" 
	"wine opt lamda: 2, opt tse: 0.625308842305" 
	"crime opt lamda:75, opt tse: 0.389023387713" 
	#[ 0.389023387713,0.625308842305,4.15967850948,5.07829980059,4.31557063032,5.54090222919,5.20591195733,4.84894305335]
	
	plt.figure(1)
	lam = range(151)
	plt.plot(lam, tr_mse[0], 'r', label = 'train crime mse')
	plt.plot(lam, te_mse[0], 'g', label = 'test crime mse')
	plt.title('crime MSE')
	plt.ylabel("mse")
	plt.legend(loc = 0)
	plt.savefig("crime.png")
	plt.clf()

	plt.figure(2)
	lam = range(151)
	plt.plot(lam, tr_mse[1], 'r', label = 'train wine mse')
	plt.plot(lam, te_mse[1], 'g', label = 'test wine mse')
	plt.title('wine MSE')
	plt.ylabel("mse")
	plt.xlabel('lambda')
	plt.legend(loc = 4)
	plt.savefig("wine.png")
	plt.clf()

	plt.figure(3)
	lam = range(151)
	plt.plot(lam, tr_mse[2], 'r', label = 'train-100-10 mse')
	plt.plot(lam, te_mse[2], 'g', label = 'test-100-10 mse')
	plt.axhline(y = 3.78, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('100-10 MSE')
	plt.ylabel("mse")
	plt.xlabel('lambda')
	plt.legend(loc = 0)
	plt.savefig("100-10.png")
	plt.clf()

	plt.figure(4)
	lam = range(151)
	plt.plot(lam, tr_mse[3], 'r', label = 'train-100-100 mse')
	plt.plot(lam, te_mse[3], 'g', label = 'test-100-100 mse')
	plt.axhline(y = 3.78, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('100-100 MSE')
	plt.ylabel('mse')
	plt.xlabel('lambda')
	plt.legend(loc = 5)
	plt.ylim([0.0, 50.0])
	plt.savefig("100-100.png")
	plt.clf()
	

	plt.figure(5)
	lam = range(151)
	plt.plot(lam, tr_mse[4], 'r', label = 'train-1000-100 mse')
	plt.plot(lam, te_mse[4], 'g', label = 'test-1000-100 mse')
	plt.axhline(y = 4.015, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('1000-100 MSE')
	plt.ylabel("mse")
	plt.xlabel('lambda')
	plt.legend(loc = 0)
	plt.savefig("1000-100.png")
	plt.clf()

	plt.figure(6)
	lam = range(151)
	plt.plot(lam, tr_mse[5], 'r', label = 'train-50(1000)-100 mse')
	plt.plot(lam, te_mse[5], 'g', label = 'test-50(1000)-100 mse')
	plt.axhline(y = 4.015, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('50(1000)-100 MSE')
	plt.ylabel("mse")
	plt.xlabel('lambda')
	plt.ylim([0.0, 20.0])
	plt.legend(loc = 2)
	plt.savefig("50(1000)-100.png")
	plt.clf()

	plt.figure(7)
	lam = range(151)
	plt.plot(lam, tr_mse[6], 'r', label = 'train-100(1000)-100 mse')
	plt.plot(lam, te_mse[6], 'g', label = 'test-100(1000)-100 mse')
	plt.axhline(y = 4.015, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('100(1000)-100 MSE')
	plt.ylabel("mse")
	plt.legend(loc = 0)
	plt.ylim([0.0, 20.0])
	plt.savefig("100(1000)-100.png")
	plt.clf()

	plt.figure(8)
	lam = range(151)
	plt.plot(lam, tr_mse[7], 'r', label = 'train-150(1000)-100 mse')
	plt.plot(lam, te_mse[7], 'g', label = 'test-150(1000)-100 mse')
	plt.axhline(y = 4.015, xmin = 0, xmax = 150, hold = None, label = 'true mse')
	plt.title('150(1000)-100 MSE')
	plt.ylabel("mse")
	plt.xlabel('lambda')
	plt.legend(loc = 0)
	plt.savefig("150(1000)-100.png")
	plt.clf()
	
main()



		

