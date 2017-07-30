

def getData(filename):
	with open(filename, 'r') as f:
		content = f.read()
		data = content.split("\n")

		return data

def train_mle(data, num_variables, num_vectors):
	p0 = 0.0 #probability Y = 0
	p1 = 0.0 #probability Y = 1
	for row in data:
		if row[num_variables] == '0':
			p0 += 1
		else:
			p1 += 1
	result = []
	for i in range(num_variables): #for each variable in the vector
		dictionary = {("0","0"):0, ("0","1"):0, ("1","0"):0, ("1", "1"):0 } #first index is x, second one is y
		for row in data: #parse the vectors
			key = (row[i], row[num_variables]) #row[num_variables] = the value of y
			dictionary[key] += 1
		for k,v in dictionary.items():
			dictionary[k] = float(v)/num_vectors
		result.append(dictionary)
	return result, p0/num_vectors, p1/num_vectors

def train_laplace(data, num_variables, num_vectors):
	p0 = 2.0 #probability Y = 0, 2 because 1 of each
	p1 = 2.0 #probability Y = 1
	for row in data:
		if row[num_variables] == '0':
			p0 += 1
		else:
			p1 += 1
	result = []
	for i in range(num_variables): #for each variable in the vector
		dictionary = {("0","0"):1, ("0","1"):1, ("1","0"):1, ("1", "1"):1 } #first index is x, second one is y
		for row in data: #parse the vectors
			key = (row[i], row[num_variables]) #row[num_variables] = the value of y
			dictionary[key] += 1
		for k,v in dictionary.items():
			dictionary[k] = float(v)/(num_vectors + 4)
		result.append(dictionary)

	return result, p0/(num_vectors + 4), p1/(num_vectors + 4)

def test(test_data, tables, p0, p1, ntest_variables, ntest_vectors):
	numcorrect_0 = 0.0
	numcorrect_1 = 0.0
	numtotal_0 = 0.0
	numtotal_1 = 0.0
	for row in test_data:
		py0 = 1.0 #probability that we classify as 0
		py1 = 1.0 #probability that we classify as 1

		for i in range(ntest_variables):
			table = tables[i] #the table of 4 values
			py0 *= (table[(row[i], "0")]/p0)
			py1 *= (table[(row[i], "1")]/p1)

		py0 *= p0
		py1 *= p1

		if py0 > py1:
			classify = "0"
		else:
			classify = "1"

		if classify == row[ntest_variables]: #match!
			if classify == "0":
				numcorrect_0 += 1
				numtotal_0 += 1
			else:
				numcorrect_1 += 1
				numtotal_1 += 1
		else:
			if row[ntest_variables] == "0":
				numtotal_1 += 1
			else:
				numtotal_0 += 1
	
	print("Class 0: tested {}, correctly classified {}".format(numtotal_0, numcorrect_0))
	print("Class 1: tested {}, correctly classified {}".format(numtotal_1, numcorrect_1))
	print("Overall: tested {}, correctly classified {}".format(numtotal_0 + numtotal_1, numcorrect_0 + numcorrect_1))
	print("Accuracy = {}".format((numcorrect_0 + numcorrect_1)/(numtotal_0 + numtotal_1)))

def run_mle():
	print("MLE: \n")
	filename = input("Enter a filename (include train):")
	data = getData(filename)

	num_variables = int(data[0])
	num_vectors = int(data[1])
	data = data[2:]
	data = [row.replace(":","").split(" ") for row in data]
	tables, p0, p1 = train_mle(data, num_variables, num_vectors)

	test_data = getData(filename.replace("train", "test"))
	ntest_variables = int(test_data[0])
	ntest_vectors = int(test_data[1])
	test_data = test_data[2:]
	test_data = [row.replace(":","").split(" ") for row in test_data]

	test(test_data, tables, p0, p1, ntest_variables, ntest_vectors)
	print("\n")

def run_laplace():
	print("Laplace: \n")
	filename = input("Enter a filename (include train):")
	data = getData(filename)

	num_variables = int(data[0])
	num_vectors = int(data[1])
	data = data[2:]
	data = [row.replace(":","").split(" ") for row in data]
	tables, p0, p1 = train_laplace(data, num_variables, num_vectors)

	test_data = getData(filename.replace("train", "test"))
	ntest_variables = int(test_data[0])
	ntest_vectors = int(test_data[1])
	test_data = test_data[2:]
	test_data = [row.replace(":","").split(" ") for row in test_data]

	test(test_data, tables, p0, p1, ntest_variables, ntest_vectors)

def main():
	run_mle()
	run_laplace()

if __name__ == '__main__':
	main()