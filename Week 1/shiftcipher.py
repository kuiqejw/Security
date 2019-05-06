# SUTD 50.020 Security Lab 1
# Simple file read in/out
#!/usr/bin/env python

# Import libraries
import sys
import argparse
import string


def simple_shift(input_stream, key,mode):
	output_stream = ""
	shift = 0
	total = len(string.printable)
	if key < 1 or key > total-1:
		print('error! String is not in stringprintable')
		return 'error'
		#string.printable includes digit, ascii_letters, punctuation and whitespace
	for i in input_stream:
		curr = string.printable.find(i)
		if mode == 'e':
			shift = key
		elif mode == 'd':
			shift = -1*key
		curr += shift
		index = curr %total
		output_stream += string.printable[index]
	return output_stream 
def printshiftcipher(filein, fileout, key, mode):
	with open(filein, mode="r", encoding='utf-8', newline='\n') as fin:
		with open(fileout, mode="w", encoding='utf-8', newline='\n') as fout:
		    text = fin.read()
		    print("a")
		    fout.write(simple_shift(text, key, mode))

def binshiftcipher(filein, fileout, key, mode):
	with open(filein, mode='rb') as fin:
		inbyte = bytearray(fin.read())
	outbyte = bytearray()

	if mode == 'e':
		for i in inbyte:
			outbyte.append((i+key)%256)
	else:
		for i in inbyte:
			outbyte.append((i+256-key)%256)
	fout = open(fileout, mode='wb')
	fout.write(outbyte)
	fin.close()
	fout.close()
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest='filein', help='input file')
	parser.add_argument('-o', dest='fileout', help = 'output file')
	parser.add_argument('-k', dest = 'key', help = 'key string', type = int)
	parser.add_argument('-m', dest='mode', help = 'd or e', type = str)
	parser.add_argument('-t', dest='typeof', help='bin or str', type=str)

	args = parser.parse_args()
	filein = args.filein
	fileout = args.fileout
	key = int(args.key)
	typeof = args.typeof
	mode = args.mode


	if key > 255 or key < 0:
		raise 'Error! key length must be between 0 and 255'
	if mode != 'd' and mode != 'e':
		raise "error! mode can only be 'e' or 'd'"
	if typeof == 'bin':
		binshiftcipher(filein, fileout, key,mode)
	elif typeof == 'str':
		printshiftcipher(filein, fileout, key,mode)
	if typeof != 'bin' and typeof != 'str':
		raise 'Error! must be bin or string'		

# simple_shift(5, 'e')

