import math
import numpy

learning_rate = 0.0001
nsteps = 10000
def getData(filename):
	with open(filename, 'r') as f:
		content = f.read()
		data = content.split("\n")

		return data

def log_likelihood(params, data):
	result = []
	for row in data:
		dot_product = numpy.dot(params, row[:-1]) #theta transpose dot x
		element = row[-1] * (math.log(1/(1 + math.exp(-1*dot_product)))) + (1 - row[-1])* (math.log(1 - (1/(1 + math.exp(-1*dot_product)))))
		result.append(element)
	return sum(result)

def train(data, num_variables, num_vectors):
	params = [0.0 for i in range(num_variables + 1)]
	print(log_likelihood(params, data))
	for i in range(nsteps):
		gradient = [0.0 for i in range(num_variables + 1)]
		for row in data:
			dot_product = numpy.dot(params, row[:-1]) #theta transpose dot x
			for j in range(num_variables + 1):				
				gradient[j] += (row[j]) * (row[-1] - (1/(1 + math.exp(-1*dot_product))))
		gradient = [elem * learning_rate for elem in gradient]
		params = [sum(x) for x in zip(gradient, params)]

	print(log_likelihood(params, data))
	return params

def test(params, test_data, ntest_variables, ntest_vectors):
	numcorrect_0 = 0.0
	numcorrect_1 = 0.0
	numtotal_0 = 0.0
	numtotal_1 = 0.0
	for row in test_data:
		dot_product = numpy.dot(params, row[:-1]) #theta transpose dot x
		p = 1.0/(1.0 + math.exp(-1 * dot_product))
		if p > 0.5:
			classify = 1.0
		else:
			classify = 0.0

		if classify == row[-1]: #match!
			if classify == 0.0:
				numcorrect_0 += 1
				numtotal_0 += 1
			else:
				numcorrect_1 += 1
				numtotal_1 += 1
		else: #no match
			if row[-1] == 0.0: #implies classify = 1
				numtotal_0 += 1
			else:
				numtotal_1 += 1

	print("Class 0: tested {}, correctly classified {}".format(numtotal_0, numcorrect_0))
	print("Class 1: tested {}, correctly classified {}".format(numtotal_1, numcorrect_1))
	print("Overall: tested {}, correctly classified {}".format(numtotal_0 + numtotal_1, numcorrect_0 + numcorrect_1))
	print("Accuracy = {}".format((numcorrect_0 + numcorrect_1)/(numtotal_0 + numtotal_1)))


def main():
	filename = input("Enter a filename (include train):")
	data = getData(filename)
	num_variables = int(data[0])
	num_vectors = int(data[1])
	data = data[2:]
	data = [row.replace(":","").split(" ") for row in data]
	data = [[float(elem) for elem in row] for row in data]
	for row in data:
		row.insert(0, 1.0)
	params = train(data, num_variables, num_vectors)
	print(params)

	test_data = getData(filename.replace("train", "test"))
	ntest_variables = int(test_data[0])
	ntest_vectors = int(test_data[1])
	test_data = test_data[2:]
	test_data = [row.replace(":","").split(" ") for row in test_data]
	test_data = [[float(elem) for elem in row] for row in test_data]
	for row in test_data:
		row.insert(0, 1.0)
	test(params, test_data, ntest_variables, ntest_vectors)




if __name__ == '__main__':
	main()