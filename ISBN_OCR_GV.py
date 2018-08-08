import io
import os
import shutil
from wand.image import Image
import argparse
import xlsxwriter
import isbnlib
import datetime

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


def create_main_folders(pdf_file,im_folder,output_folder,debug):
    '''
    Creates all the folders required to store the intermediate and final output
    '''
    dir_list = []
    file_name = os.path.splitext(pdf_file)[0]
    dir_list.append(file_name)
    im_dir = os.path.join(file_name,im_folder)
    dir_list.append(im_dir)
    if debug is True:
        output_dir = os.path.join(file_name,output_folder)
        dir_list.append(output_dir)
    for directory in dir_list:
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            if debug is True:
                print('Folder: '+(directory)+ ' created.')
        except OSError:
            if debug is True:
                print ('Error: Creating directory' +  directory)

def convert_pdf(pdf_file,im_folder,resolution,debug):
    '''
    Converts pdf to png format, separating individual pages and store
    them in folder named as pdf file name
    '''
    directory = os.path.splitext(pdf_file)[0]
    image_dir = os.path.join(directory,im_folder)

    #writes image using wand
    with(Image(filename=pdf_file,resolution=resolution)) as source:
        images=source.sequence
        pages=len(images)
        for i in range(pages):
            Image(images[i]).save(filename=os.path.join(image_dir,str(i)+'.png'))
    if debug is True:
        print('File conversion complete!')

def get_like_isbn(string,isbn_check):
    '''
    Takes in string and returns ISBN-like string
    Returns None if there is no ISBN
    '''
    string = string.replace('\n','')
    index = string.find(isbn_check)
    if string=='' or index ==-1:
        pass
    else:
        like_isbn = string[index:index+13]
        return(like_isbn)

#ISBN check sum  - returns a list containing flags for validity
def verify_isbn(like_isbn_ls):
    isbn_flagged = []
    for isbn in like_isbn_ls:
        valid = True
        isbn_digit_ls = []
        #check 1: all characters are digits
        for digit in isbn:
            try:
                isbn_digit_ls.append(int(digit))
            except ValueError:
                valid = False
                isbn_flagged.append(('Invalid - non-number',isbn))
                break
        #checks for validity of isbn using check sum
        if valid is True:
            digit_sum = 0
            index_max = int(0.5*(len(isbn_digit_ls)-1))
            try:
                for i in range(index_max):
                    digit_sum += isbn_digit_ls[2*i] + 3*isbn_digit_ls[2*i+1]
                digit_sum += isbn_digit_ls[12]
                if digit_sum%10 == 0:
                    isbn_flagged.append(('Valid',isbn))
                else:
                    isbn_flagged.append(('Invalid - wrong digit(s)',isbn))
            except IndexError:
                isbn_flagged.append(('Invalid - missing digit(s)',isbn))
    return(isbn_flagged)

#Excel functions
def write_to_excel(worksheet,isbn_ls):
    create_column_titles(worksheet)
    row = 1
    for validity,isbn in isbn_ls:
        worksheet.write(row,0,validity)
        worksheet.write(row,1,isbn)
        if validity=='Valid':
            try:
                book = isbnlib.meta(isbn,service='default',cache='default')
                authors = ','.join(book['Authors']) #to concatenate the list of authors
                worksheet.write(row,2,book['Title'])
                worksheet.write(row,3,authors)
                worksheet.write(row,4,book['Publisher'])
                worksheet.write(row,5,book['Year'])
                worksheet.write(row,6,book['Language'])
            except:
                worksheet.write(row,2,'No info found!')
        row +=1


def create_column_titles(worksheet):
    worksheet.write(0,0,'Validity')
    worksheet.write(0,1,'ISBN')
    worksheet.write(0,2,'Title')
    worksheet.write(0,3,'Author')
    worksheet.write(0,4,'Publisher')
    worksheet.write(0,5,'Year')
    worksheet.write(0,6,'Language')


def recognise_text(pdf_file_name,im_file,im_file_name,im_folder,output_folder,isbn_check,debug):
    '''
    Calls an API to Google Vision to recognise text
    '''
    im_file_dir = os.path.join(pdf_file_name,im_folder,im_file)
    text_annotations = google_vision(im_file_dir)
    like_isbn_ls = []
    original_text_ls = []

    for text in text_annotations:
        string = text.description
        original_text_ls.append(string)
        like_isbn = get_like_isbn(string,isbn_check)

        if like_isbn is None or like_isbn in like_isbn_ls:
            pass
        else:
            like_isbn_ls.append(like_isbn)

    if debug is True:
        with open(os.path.join(pdf_file_name,output_folder,f'{im_file_name}_text.txt'),'w') as f:
            f.write(f'This is the original:{original_text_ls} \n This is the like_isbn:{like_isbn_ls}')

    return like_isbn_ls

def google_vision(im_file_dir):
    '''
    Passes in image directory, returns text annotations
    '''
    #Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(im_file_dir, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.text_detection(image=image)
    return response.text_annotations

if __name__ == '__main__':
    print(datetime.datetime.now())
    #key parameters
    isbn_check = '978' #used for checking validity of isbn
    resolution=300
    debug = False

    #folder names
    im_folder = 'images'
    output_folder = 'output'

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Save intermediate files in folders for debugging.", action="store_true")
    parser.add_argument("pdf_file", type=str, help="Select a file to extract ISBN from. Format: 'example.pdf'")
    parser.add_argument("key_dir", type=str, help="Provide the path to the authentication credentials. Format:'PATH'")
    args = parser.parse_args()

    if args.debug:
        debug = True
        print("Debug mode turned on.")

    pdf_file = args.pdf_file
    pdf_file_name = os.path.splitext(pdf_file)[0]
    key_dir = args.key_dir

    #set environment
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_dir
    #File conversion
    create_main_folders(pdf_file,im_folder,output_folder,debug)
    convert_pdf(pdf_file,im_folder,resolution,debug)

    image_file_ls = os.listdir(os.path.join(pdf_file_name,im_folder))

    #Pass image into Google Vision and writes to Excel
    workbook = xlsxwriter.Workbook(os.path.join(pdf_file_name,'results.xlsx'))

    for im_file in image_file_ls:
        im_file_name = os.path.splitext(im_file)[0]
        worksheet = workbook.add_worksheet(im_file_name)
        like_isbn_ls = recognise_text(pdf_file_name,im_file,im_file_name,im_folder,output_folder,isbn_check,debug)
        verified_isbn_ls = verify_isbn(like_isbn_ls)
        write_to_excel(worksheet,verified_isbn_ls)

    workbook.close()

    print(datetime.datetime.now())
