import matplotlib.pyplot as plt



plt.figure(1)
x = range(5)
train_splits = [128.0, 64.0, 16.0, 4.0, 1.0]
mle_perpelxity = [274677.2772400548, 61904.059283452778, 11356.209581092513, 8867.029411472573, 8612.3464106221036]
map_perpelxity = [10106.876172886021, 10004.357699410373, 9338.5963276938328, 8800.5378030988559, 8609.536150969845]
pre_perplexity = [9812.3779694666464, 9677.3356439498857, 9191.1148892757865, 8779.9152145737517, 8607.9713604658882]
plt.plot(x, mle_perpelxity, 'r', marker = '*', label = 'MLE perplexity')
plt.plot(x, map_perpelxity, 'g', marker = '*', label = 'MAP perplexity')
plt.plot(x, pre_perplexity, 'b', marker = '*', label = 'pred perplexity')
plt.xticks(x, map(lambda x: r'$\frac{N}{%s}$'%str(x), train_splits))
plt.ylabel("test set perplexity")
plt.legend()
plt.show()



plt.figure(2)
plt.subplot(211)
alpha_prime = alpha_prime = range(1,11)
log_evidence = [-46056.263237181833, -45979.245905323871, -45976.858865881935, -45983.171410863055, -45990.078837020381, -45996.20978735041, -46001.423333142622, -46005.828072882112, -46009.565792742847, -46012.762308460646]
pre_perplexity_2 = [10097.789476896151, 9809.3643839545348, 9780.7190111418313, 9790.4656099736621, 9806.9346732766044, 9823.3215568999403, 9838.0697908151778, 9850.9752475528039, 9862.1959090876644, 9871.9664174391801]

plt.plot(alpha_prime, log_evidence, 'r', marker = 'o', label = 'log evidence')
plt.ylabel('log evidence')
plt.legend()
plt.subplot(212)
plt.plot(alpha_prime, pre_perplexity_2, 'g', marker = '*', label = 'perplexity')
plt.ylabel("perplexity")
plt.legend()
plt.show()

	