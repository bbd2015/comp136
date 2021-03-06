import matplotlib.pyplot as plt
import numpy as np

x = range(10)
A_ge = [ 0.1958959,   0.12562563,  0.09884885,  0.08758759,  0.07427427,  0.06906907,
  0.06401401,  0.06416416,  0.05855856,  0.0545045 ]
A_ge_std = [ 0.02639608, 0.01587305,  0.01288193,  0.01189466,  0.01200408,  0.00998997,
  0.00968038,  0.01077832,  0.00967663,  0.00839723]
A_di = [ 0.45205205,  0.42637638,  0.41561562, 0.4032032,   0.40345345,  0.38963964,
  0.3950951,   0.39379379,  0.38768769,  0.38663664]
A_di_std = [ 0.02136475,  0.02224549,  0.01862265,  0.01819125,  0.02241264,  0.02170609,
  0.01340594,  0.01695183,  0.01704673,  0.01733243]


B_ge = [0.36010101,  0.28939394, 0.25050505, 0.20909091,  0.22575758,  0.21464646,
  0.19494949,  0.19393939, 0.18939394,  0.18181818]
B_ge_std = [ 0.08414966,  0.06372649,  0.05005863,  0.06202876,  0.04683918, 0.04233095,
  0.04975196,  0.04328138,  0.04591517,  0.03648968]
B_di = [ 0.17424242 , 0.1969697,   0.1959596,   0.16464646,  0.16969697,  0.16565657,
  0.15858586,  0.15858586,  0.15353535,  0.1479798 ]
B_di_std = [ 0.05331812,  0.03931616,  0.04026491,  0.04041666,  0.0367821,   0.03317993,
  0.03685139,  0.03048765,  0.03404505,  0.03312223]

USPS_ge =[ 0.49746589,  0.48479532,  0.20318389,  0.11708902,  0.08434048,  0.06920078,
  0.06133853, 0.05841455,  0.05302144, 0.05022742]
USPS_ge_std = [ 0.04095238,  0.04116446,  0.03153513,  0.01730348,  0.00987483, 0.01088539,
  0.01087355,  0.0072847 ,  0.00893857,  0.00707656]
USPS_di = [ 0.06185835,  0.05490578,  0.05373619 , 0.04593892,  0.04476933,  0.04522417,
  0.04320988,  0.04320988,  0.03983106 , 0.03853151]
USPS_di_std = [ 0.00891776 , 0.01296048,  0.00920776 , 0.00877097,  0.00833864,  0.00863587,
  0.00925441,  0.00808569,  0.00796892,  0.00749382]

plt.figure(1)
plt.errorbar(x, A_ge, A_ge_std,  marker = '^')
plt.errorbar(x, A_di, A_ge_std,  marker = '*')
plt.title('A data set')
plt.xlabel('train size')
plt.ylabel('error')
plt.legend(['generative', 'discriminative'], loc = 0)
plt.savefig("A.png")
plt.clf()


plt.figure(2)
plt.errorbar(x, B_ge, B_ge_std,  marker = '^')
plt.errorbar(x, B_di, B_ge_std,  marker = '*')
plt.title('B data set')
plt.xlabel('train size')
plt.ylabel('error')
plt.legend(['generative', 'discriminative'], loc = 0)
plt.savefig("B.png")
plt.clf()


plt.figure(3)
plt.errorbar(x, USPS_ge, USPS_ge_std,  marker = '^')
plt.errorbar(x, USPS_di, USPS_ge_std,  marker = '*')
plt.title('USPS data set')
plt.xlabel('train size')
plt.ylabel('error')
plt.legend(['generative', 'discriminative'], loc = 0)
plt.savefig("USPS.png")
plt.clf()


