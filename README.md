# ISBN-identification
A project on identifying International Standard Book Number from scanned invoices

There are two implementations:
1. Google Vision (ISBN_OCR_GV.py)
-This is the script for the implementation using Google Vision. 
-Run it from the command line in Linux using Python in the format:
python ISBN_OCR_GV.py "pdf_file" "key_directory"
-Make sure you have the following packages installed in Python:
Wand
xlsxwriter
isbnlib
