import sys
import argparse

def read_flag(flag):
	try:
		byte = open(flag, mode='rb').read()
	except FileNotFoundError:
		print('Cant find the input')
	print(type(byte))
	return byte
def binshiftcipher(inbyte, fileout, key):
	outbyte = bytearray()
	for i in inbyte:
		outbyte.append((i+key)%256)
	fout = open(fileout, mode='wb')
	fout.write(outbyte)
	fout.close()

def decrypt_flag(flag):
	flag_b = read_flag(flag)
	for key in range(1,256):
		name = 'flag_capture' + str(key)
		binshiftcipher(flag_b, name, key)
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('flag')
	args = parser.parse_args()
	flag = args.flag

	decrypt_flag(flag)