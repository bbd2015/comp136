import matplotlib.pyplot as plt
import datetime

a_newton_time = [ 0.01537367,  0.02855167,  0.04108567,  0.05452367,  0.068228  ]
a_newton_error = [0.05855855855855856, 0.04804804804804805, 0.046546546546546545, 0.046546546546546545, 0.046546546546546545]
a_ga_time = 
a_ga_error = 


usps_newton_time = [ 0.02759267,  0.05277067 , 0.078221 ,   0.10310333,  0.12794833 , 0.15363867,
  0.17856  ,   0.204489 ,   0.22854267]
usps_newton_error = [0.04093567251461988, 0.042884990253411304, 0.03313840155945419, 0.037037037037037035, 0.03898635477582846, 0.037037037037037035, 0.03313840155945419, 0.03313840155945419, 0.03508771929824561]
usps_ga_time = 
usps_ga_error = 


plt.figure(1)
plt.plot(a_newton_time, a_newton_error)
plt.plot(a_ga_time, a_ga_error)
plt.xlabel('time line')
plt.ylabel('error rate')
plt.title('gradient A data')
plt.gcf().autofmt_xdate()
plt.savefig('A task2.png')
plt.clf()

plt.figure(2)
plt.plot(usps_newton_time, usps_newton_error)
plt.plot(usps_ga_time, usps_ga_error)
plt.xlabel('time line')
plt.ylabel('error rate')
plt.title('gradient A data')
plt.gcf().autofmt_xdate()
plt.savefig('USPS task2.png')
plt.clf()