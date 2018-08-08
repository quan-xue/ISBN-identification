# ISBN-identification
A project on identifying International Standard Book Number from scanned invoices

Both scripts take in a pdf file of a scanned invoice and outputs an excel file containing the ISBNs and the books' information in the invoices.

There are two implementations:\
A. Google Vision (ISBN_OCR_GV.py) \
-This is the script for the implementation using Google Vision. \
-Run it from the command line in Linux using Python in the format: \
python ISBN_OCR_GV.py "pdf_file.pdf" "key_directory.json" \
-Make sure you have the following packages installed in Python:
1. Wand 
2. xlsxwriter 
3. isbnlib 
4. API key for Google Vision
Instructions for Google Vision can be found here: 
https://cloud.google.com/vision/docs/libraries

B. Tesseract + CTPN 
-This script uses the CTPN package described by: \
Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
Connectionist Text Proposal Network, ECCV, 2016. \
https://github.com/tianzhi0549/CTPN
-Please make sure to install the relevant packages by following the isntructions described in the CTPN package.\ 


asd
