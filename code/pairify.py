
import sys

if __name__ == "__main__":
	file1=sys.argv[1]
	file2=sys.argv[2]
	posfile=sys.argv[3]
	with open(file1) as f1:
		arr1 = [line.strip() for line in f1]

	with open(file2) as f2:
		arr2 = [line.strip() for line in f2]

	with open(posfile) as f3:
		pos = [line.strip() for line in f3]
		pos = "\t".join(pos)

	ctr = 1
	for l1 in arr1:
		for l2 in arr2:
			ppstr=format("\"%s\",\"%s\"" % (l1,l2))
			if ppstr not in pos:
				print(format("%d,%s,%s" % (ctr,l1,l2)))
				ctr = ctr+1
