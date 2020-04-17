
import sys

if __name__ == "__main__":
	file1=sys.argv[1]
	file2=sys.argv[2]
	with open(file1) as f1:
		arr1 = [line.strip() for line in f1]

	with open(file2) as f2:
		arr2 = [line.strip() for line in f2]

	for l1 in arr1:
		for l2 in arr2:
			print(l1,l2)
