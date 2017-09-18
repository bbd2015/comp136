# comp136 project2 Task 2
# beibei du
# 10/27/2016

import os
import sys
import math
import numpy as np 
from numpy.linalg import inv
import random
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

def main():
	tr = read_file('train-1000-100.csv')
	tr_r = read_file_r('trainR-1000-100.csv')
	te = read_file('test-1000-100.csv')
	te_r = read_file_r('testR-1000-100.csv')
	lam_val = [15, 27, 50]
	train_size = [10, 20,30, 40,50, 100, 150, 200, 250, 300,350, 400,450, 500,550, 600,650, 700,750,800]
	lam_mse_list = [] 
	tr = tr.tolist()
	for lam in lam_val:
		lam_mse = []
		for i in train_size:
			te_mse_list = []
			for j in range(10):
				train = random.sample(tr, i)
				train_r = []
				for item in train:
					index = tr.index(item)
					train_r.append(tr_r[index])
				train = np.array(train)
				w = calculate_w(lam, train, train_r)
				te_mse = calculate_mse(w, te, te_r)
				te_mse_list.append(te_mse)
			te_mse_avg = sum(te_mse_list) / float(10)
			lam_mse.append(te_mse_avg)
		print "lamda = %f"%lam
		print lam_mse
		lam_mse_list.append(lam_mse)

	plt.figure(1)
	x = range(20)
	plt.plot(x, lam_mse_list[0], 'r', marker = '*', label = 'lambda = 15')
	plt.plot(x, lam_mse_list[1], 'g', marker = '*', label = 'lambda = 27')
	plt.plot(x, lam_mse_list[2], 'b', marker = '*',label = 'lambda = 50')
	plt.axhline(y = 4.015, xmin = 0, xmax = 150, hold = None, label = 'True mse')
	plt.xticks(x, map(lambda x: r'$\frac{%s}{N}$'%str(x), train_size))
	plt.title('Learning Curves')
	plt.ylabel("mse")
	plt.xlabel('train size')
	plt.ylim(3.5,12.0)
	plt.legend(loc = 1)
	plt.show()
main()





