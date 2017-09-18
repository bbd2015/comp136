
filename = 'index.csv'

result = []
with open(filename, 'r') as f:
	for line in f:
		line = line.strip('\n').split(',')
		result.append(line[1])
print result


