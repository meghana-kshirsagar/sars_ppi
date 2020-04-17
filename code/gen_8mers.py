
import numpy as np
import sys

if __name__ == "__main__":
	infile = sys.argv[1]
	with open(infile,'r') as fin:
		for line in fin:
			line = line.strip()
			for k in range(len(line)-8):
				print(line[k:k+8])
	
