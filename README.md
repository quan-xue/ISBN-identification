# ISBN-identification
A project on identifying International Standard Book Number from scanned invoices

Both scripts take in a pdf file of a scanned invoice and outputs an excel file containing the ISBNs and the books' information in the invoices.

There are two implementations:
### A. Google Vision (ISBN_OCR_GV.py) 
This is the script for the implementation using Google Vision. \
Make sure you have the following packages installed in Python:
1. Wand 
2. xlsxwriter 
3. isbnlib 
4. API key for Google Vision
Instructions for setting up Google Vision can be found here: 
https://cloud.google.com/vision/docs/libraries
#### How to run the script
Store your key in the same directory as the script.
Run it from the command line in Linux using Python in the format: 
$ python ISBN_OCR_GV.py [-d] "pdf_file.pdf" "key_directory.json" 
There is an optional debug mode, activated by -d or --debug.

### B. Tesseract + CTPN 
This script uses the CTPN package for the detection of text boxes, as described by: 
Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
Connectionist Text Proposal Network, ECCV, 2016. 
https://github.com/tianzhi0549/CTPN
Please make sure to install the relevant packages by following the instructions described in the CTPN package.\ 

#### How to run the script
Paste the script into the folder 'ctpn' after you have setup CTPN.
Run the script in Linux using Python from the parent directory in the format:
$ python ./ctpn/ISBN_OCR.py "pdf_file.pdf" "key_directory.json" angle_of_rotation
*Angle of rotation would the required clock-wise rotation for the pdf file eg. 0, 90, 180, 270

There is an optional debug mode, activated by -d or --debug.



