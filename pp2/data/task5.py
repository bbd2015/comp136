# figure analysi for task3, task4, task5

import matplotlib.pyplot as plt
import numpy as np 

s = ['crime', 'wine', '100-10', '100-100', '1000-100', '50(1000)-100', '100(1000)-100', '150(1000)-100']
# part 1 test mse
y1 =[ 0.389023387713,0.625308842305,4.15967850948,5.07829980059,4.31557063032,5.54090222919,5.20591195733,4.84894305335]
lambda_1 = [75,2,8,22,27,8,19,23]
# cross validation mse
y2 = [0.39227040752452375, 0.62593622669660731, 4.1757091596711433, 5.0808888179185328, 4.3227223508824659, 5.9344653841860655, 5.2599827410154809, 4.9341926077600835]
lambda_2 = [150,2,12,20,39,24,30,46]
# model selection mse
y3 = [0.39110220806201251, 0.6267448249308214, 4.1801330312023159, 7.3525369250732879, 4.3383499509291168, 5.7895752937487694, 5.7339307345248693, 5.2489966154820733]

x = range(8)
plt.figure(1)
plt.plot(x, y1, 'r', marker = '*')
plt.plot(x, y2, 'g', marker = 's')
plt.plot(x, y3, 'b', marker = 'o')
plt.xticks(x, s, rotation = 10)
plt.legend(['regularization', 'cross validation', 'model selection'], loc = 2)
plt.ylabel('MSE')
plt.xlabel('data set')
plt.show()

plt.figure(8)
diff = list(np.array(y3) - np.array(y2))
plt.plot(x, diff, 'r', marker = '*')
plt.xticks(x, s, rotation = 10)
plt.legend(['model selection - cross validation'], loc = 2)
plt.ylabel('MSE difference')
plt.xlabel('data set')
plt.show()

# task 3 compare MSE of part1 and cross validation
plt.figure(2)
plt.plot(x, y1, 'r', marker = '*')
plt.plot(x, y2, 'g', marker = 's')
plt.xticks(x, s, rotation = 10)
plt.legend(['regularization', 'cross validation'], loc = 2)
plt.ylabel('MSE')
plt.xlabel('data set')
plt.show()


# Task 3 in terms of lambda
plt.figure(3)
plt.plot(x, lambda_1, 'r', marker = '*')
plt.plot(x, lambda_2, 'g', marker = 's')
plt.xticks(x, s, rotation = 10)
plt.legend(['regularization', 'cross validation'], loc = 2)
plt.ylabel('Optimal lambda')
plt.xlabel('data set')
plt.show()

# Task 3 in fixed number of features
plt.figure(4)
x1 = range(5)
s1 = ['50(1000)-100', '100(1000)-100', '150(1000)-100', '1000-100', '100-100']
y1_1 =[ 5.54090222919,5.20591195733,4.84894305335,4.31557063032,5.07829980059]
y2_1 = [5.9344653841860655, 5.2599827410154809, 4.9341926077600835,4.3227223508824659,5.0808888179185328]
plt.plot(x1, y1_1, 'r', marker = '*')
plt.plot(x1, y2_1, 'g', marker = 's')
plt.xticks(x1, s1, rotation = 10)
plt.legend(['regularization', 'cross validation'], loc = 0)
plt.ylabel('TEST MSE')
plt.xlabel('data set')
plt.show()

plt.figure(5)
y1_lambda = [8,19,23,27,22]
y2_lambda = [24,30,46,39, 20]
plt.plot(x1, y1_lambda, 'r', marker = '*')
plt.plot(x1, y2_lambda, 'g', marker = 's')
plt.xticks(x1, s1, rotation = 10)
plt.legend(['regularization', 'cross validation'], loc = 0)
plt.ylabel('lambda')
plt.xlabel('data set')
plt.show()


# Task4 compare MSE of model selection and part1
plt.figure(6)
plt.plot(x, y1, 'r', marker = '*')
plt.plot(x, y3, 'g', marker = 's')
plt.xticks(x, s, rotation = 10)
plt.legend(['regularization', 'model selection'], loc = 2)
plt.ylabel('MSE')
plt.xlabel('data set')
plt.show()

# Task4 fixed number of features
plt.figure(7)
x1 = range(5)
s1 = ['50(1000)-100', '100(1000)-100', '150(1000)-100', '1000-100', '100-100']
y1_1 =[ 5.54090222919,5.20591195733,4.84894305335,4.31557063032, 5.07829980059]
y3_1 = [5.7895752937487694, 5.7339307345248693, 5.2489966154820733,4.3383499509291168,7.3525369250732879]
plt.plot(x1, y1_1, 'r', marker = '*')
plt.plot(x1, y3_1, 'g', marker = 's')
plt.xticks(x1, s1, rotation = 10)
plt.legend(['regularization', 'model selection'], loc = 0)
plt.ylabel('TEST MSE')
plt.xlabel('data set')
plt.show()

