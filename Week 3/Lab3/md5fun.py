import hashlib
import random
import time
from string import ascii_lowercase, digits
from itertools import permutations


#read and write
def getFileContent(filename):
	file = open(filename, mode = 'r', encoding='utf-8', newline = '\n')
	lines = []
	for line in file:
		if line.strip():
			lines.append(line.strip())
	return lines

def getFileContentDict(filename):
	file = open(filename, mode = 'r', encoding='utf-8', newline = '\n')
	lines = []
	for line in file:
		if line.strip():
			lines.append(line.strip())
	return lines

def writeFile(filename, content):
	#content in list
	file = open(filename, mode = 'w', encoding = 'utf-8', newline = '\n')
	file.write(content)
	print(content)
	file.close()
	return

def printList(list):
	out = ''
	for i in list:
		out += i + '\n'
	return out




#brute force

def individualElement(words):
	s = set()
	for i in range(len(words)):
		x = ''.join(sorted(set(words[i])))
		s.add(x)
	print(len(s))
	return s


def getPasswords(words, hashes):
	passwords = []
	s = set()#set to try
	#attempt all available words first
	for i in words:
		hashed = hashlib.md5(i.encode('utf-8')).hexdigest()
		if hashed in hashes:
			passwords.append(i)
			hashes.remove(hashed)
	passwords = editedBruteForce(hashes, passwords)
	writeFile('pass2.txt', printList(passwords))

def editedBruteForce(hashes, passwords):
	string = 'abcdefghijklmnopqrstuvwxyz0123456789'
	len_charac= len(string)
	for i in range(len_charac):
		for j in range(len_charac):
			for x in range(len_charac):
				for y in range(len_charac):
					for z in range(len_charac):
						word = string[i] + string[j] + string[x]+string[y]+string[z]
						e = hashlib.md5(word.encode('utf-8')).hexdigest()
						if e in hashes:
							passwords.append(word)
							hashes.remove(e)
							if len(passwords) == 15:
								return passwords
	return passwords

def bruteForce(hashes):
	string = 'abcdefghijklmnopqrstuvwxyz0123456789'
	len_charac= len(string)
	passwords = []
	for i in range(len_charac):
		for j in range(len_charac):
			for x in range(len_charac):
				for y in range(len_charac):
					for z in range(len_charac):
						word = string[i] + string[j] + string[x]+string[y]+string[z]
						e = hashlib.md5(word.encode('utf-8')).hexdigest()
						if e in hashes:
							passwords.append(word)
							hashes.remove(e)
							if len(passwords) == 15:
								return passwords
	return passwords

def Part3():
	startTime = time.perf_counter()
	passwords = bruteForce(hash5)
	endTime = time.perf_counter()
	print(passwords)
	print('Elapsed time = ', endTime - startTime)
	writeFile('pass5.txt', printList(passwords))
	return

def salting():
	characters = ascii_lowercase
	salt = characters[int(random.random()*100) %len(characters)]
	return salt

def Part5():
	passwords = getFileContent('pass5.txt')
	passwordsSalted = []

	outPass6 = ''
	outSalt = ''

	for p in passwords:
		salt = salting()
		passwordsSalted.append((p, salt))
		newPassword = p + salt
		outPass6 += newPassword + '\n'
		outSalt  += hashlib.md5(newPassword.encode('utf-8')).hexdigest() +'\n'
	writeFile('pass6.txt', outPass6)
	writeFile('salted6.txt',outSalt)

def Part6():
	hashesContent = getFileContent('hashes.txt')
	hashes = []
	difficulty = -1
	for line in hashesContent:
		if 'Weak' in line or 'Moderate' in line or 'Strong' in line:
			hashes.append([])
			difficulty += 1
			hashes[difficulty] = []
		else:
			print(difficulty)
			hashes[difficulty].append(line)
	writeFile('hashes_weak.txt', printList(hashes[0]))
	writeFile('hashes_moderate.txt', printList(hashes[1]))
	writeFile('hashes_strong.txt', printList(hashes[2]))

	return

if __name__=="__main__":
    # parser=argparse.ArgumentParser(description='Brute force.')
    # parser.add_argument('-i', dest='infile',help='Input file')
    # parser.add_argument('-d', dest='dictfile',help='Dictionary file')
    # parser.add_argument('-o', dest='outfile',help='Output file')

    # args=parser.parse_args()
    # infile=args.infile
    # dictfile=args.dictfile
    # outfile=args.outfile

    # if infile==None or outfile==None or dictfile==None:
    #     print 'Missing infile, outfile, or dictfile'
    #     printusage();
    #     sys.exit(1)

    # print 'Reading from: ',infile
    # print 'Dictionary from: ',dictfile
    # print 'Writing to: ',outfile

	words5 = getFileContent('words5.txt')
	hash5 = getFileContent('hash5.txt')
	# start_time = time.time()
	# print('start')
	# Part3()
	# print('---%s seconds ----' %(time.time() - start_time))
	#230.76921701 seconds
	start_time = time.time()
	print('start', start_time)
	getPasswords(words5, hash5)
	print('---%s seconds ----' %(time.time() - start_time))
	# start_time = time.time()
	# print('start')
	# Part6()
	# print('---%s seconds ----' %(time.time() - start_time))
	#0.04687643 seconds

	# getPasswords(words5, hash5)