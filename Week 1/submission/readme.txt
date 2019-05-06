# To test shift cipher in text mode
python3 shiftcipher.py -i sherlock.txt -o str_en.out -k 4 -m e -t str
python3 shiftcipher.py -i str_en.out -o str_de.out -k 4 -m d -t str

# To test shift cipher in binary mode
python3 shiftcipher.py -i sherlock.txt -o bin_en.out -k 30 -m e -t bin
python3 shiftcipher.py -i bin_en.out -o bin_de.out -k 30 -m d -t bin

# To decrypt flag
python3 flag.py flag