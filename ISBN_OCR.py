from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
from wand.image import Image
import math
import pytesseract
import xlsxwriter
import isbnlib
import json
import datetime
import argparse

from natsort import natsorted
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

def create_main_folders(pdf_file,im_folder,coord_folder,output_folder,
                        crop_boxes_folder,debug):
    '''
    Creates all the folders required to store the intermediate and final output
    '''
    dir_list = []
    file_name = os.path.splitext(pdf_file)[0]
    dir_list.append(file_name)
    im_dir = os.path.join(file_name,im_folder)
    dir_list.append(im_dir)
    coord_dir = os.path.join(file_name,coord_folder)
    dir_list.append(coord_dir)
    if debug is True:
        output_dir = os.path.join(file_name,output_folder)
        dir_list.append(output_dir)
        crop_boxes_dir = os.path.join(file_name,crop_boxes_folder)
        dir_list.append(crop_boxes_dir)
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
    Converts pdf to png format, separating individual pages and storing them in folder named as pdf file name
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

def rotate(image_dir,angle,rotate):
    '''
    function for rotation (calculate clockwise angle for rotation:[90,180,270])
    reads directory for image, replaces image in original folder with rotated image
    '''
    #reads image as grayscale
    image = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)
    num = round(angle/90) #how many rounds of clockwise rotation to do
    rotated = image
    for i in range(num):
        temp_image=cv2.transpose(rotated)
        rotated=cv2.flip(temp_image,flipCode=1) #rotate clockwise
    if debug is True:
        print("Image rotated by: {:.3f}".format(angle))
    return rotated


def convert_rotate(pdf_file,im_folder,coord_folder,output_folder,debug,resolution=300,angle=0):
    '''
    merges the two components of step 1 - converting from pdf to separate image pages
    2 - rotating the pages
    '''
    create_main_folders(pdf_file,im_folder,coord_folder,output_folder,crop_boxes_folder, debug)
    convert_pdf(pdf_file,im_folder,resolution,debug)
    pdf_file_name = os.path.splitext(pdf_file)[0]
    image_ls = os.listdir(os.path.join(pdf_file_name,im_folder))
    for image in image_ls:
        image_file_name = os.path.join(pdf_file_name,im_folder,image)
        #assumes all images are rotated in the same way
        rotated_image = rotate(image_file_name,angle,rotate)
        deskewed_image = deskew(rotated_image)
        cv2.imwrite(image_file_name,deskewed_image)
    if debug is True:
        print('***********************PDF conversion and rotation successful!***********************')

#function for deskewing, returns rotated image
def deskew(image):
    # convert the image to grayscale and flip the foreground and background to ensure foreground 
    #is now "white" and the background is "black"
    gray = cv2.bitwise_not(image)

    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that are greater than zero, then use these coordinates tocompute a rotated 
    # bounding box that contains all coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
    if angle < -45: angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

def draw_ctpn_boxes(img,image_name,boxes,scale,pdf_file_name,coord_folder,debug):
    base_name = image_name.split('/')[-1]
    with open(os.path.join(pdf_file_name,coord_folder,'res_{}.txt'.format(base_name.split('.')[0])), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line) #output results -

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    if debug is True:
    	cv2.imwrite(os.path.join(pdf_file_name,base_name), img)

def ctpn(sess, net, image_name,pdf_file_name,coord_folder,debug):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector() #author's algorithms
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_ctpn_boxes(img, image_name, boxes, scale,pdf_file_name,coord_folder,debug)
    timer.toc()
    if debug is True:
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))


#This part of the program:
#1. Merges the text boxes identified by CTPN
#2. Crops out the text boxes based on either area or rank thresholds

def readfile(file):
    '''
    Reads opp corner coordinates line-by-line from text file,
    returning them in tuples within list
    e.g. [(1848, 530, 1980, 553),...,(1584, 394, 1848, 427)]
    '''
    assert file.endswith('.txt')
    coordinates_ls = []
    with open(file,'r') as f:
        for line in f:
            line = line.rstrip()
            coordinates = tuple(map(int,line.split(sep=',')))
            coordinates_ls.append(coordinates)
    return coordinates_ls


#the centroids are required for comparing distances and making merging decisions later
def gen_cen_list(ls):
    '''
    Generates list centroids [(x1, y1),...,(xn, yn)] from a list of opp corners
    Uses function gen_centroid
    '''
    centroids_ls = []
    for opp_corner_coords in ls:
        min_x,min_y,max_x,max_y=opp_corner_coords
        x = (min_x+max_x)/2
        y= (min_y+max_y)/2
        centroids_ls.append((x,y))
    return centroids_ls

def centroid_shortlist(cen,cen_ls,thresh):
    '''
    Finds distance between current centroid and every other centroid in list and
    filters out short-listed centroids. Returns two lists (c1,c2) and (distance,c1,c2)
    Centroid must be within distance threshold to be selected for merging
    '''
    x1,y1 = cen
    centroid_shortlist = []
    distance_shortlist = []
    for i in cen_ls:
        x2,y2 = i
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        if distance<thresh and distance!=0:
            centroid_shortlist.append((x2,y2))
            distance_shortlist.append((distance,x2,y2))
    return centroid_shortlist,distance_shortlist

def merge_boxes(centroid,distance_shortlist,cen_minmax_dict):
    '''
    Takes in centroid coordinates and the other boxes shortlisted for merging,
    outputs merged box in opp. corner coordinates.
    '''
    #Find min-max coords for merging later of centroid being compared
    b1_c1,b1_c2 = centroid
    b1_x1,b1_y1,b1_x2,b1_y2 = cen_minmax_dict[b1_c1,b1_c2]
    b1 = (b1_x1,b1_y1,b1_x2,b1_y2)

    #Merge from nearest box onwards
    to_merge_ls = sorted(distance_shortlist,key=lambda tup: tup[0],reverse=True)
    to_merge_ls_minmax = []

    #Converts (distance,centroid) to (distance,min max)
    for i in to_merge_ls:
        __,b2_c1,b2_c2 = i
        b2_x1,b2_y1,b2_x2,b2_y2 = cen_minmax_dict[b2_c1,b2_c2]
        to_merge_ls_minmax.append((b2_x1,b2_y1,b2_x2,b2_y2))

    #initliases new box
    new_box = b1
    #loops over short-listed centroids starting from shortest distance
    while len(to_merge_ls_minmax)>0:
        last = len(to_merge_ls_minmax)-1
        b2_x1,b2_y1,b2_x2,b2_y2 = to_merge_ls_minmax[last]
        #Find newly merged box
        nw_x = min(new_box[0],b2_x1)
        nw_y = min(new_box[1],b2_y1)
        se_x = max(new_box[2],b2_x2)
        se_y = max(new_box[3],b2_y2)
        new_box = (nw_x,nw_y,se_x,se_y)
        del to_merge_ls_minmax[last]
    return new_box

#debugging function

def draw_box(image,coordinates_ls):
    '''
    Draws boxes using min-max coods. Returns image with boxes drawn.
    '''
    boxed_image = image.copy()
    for coordinates in coordinates_ls:
        min_x,min_y,max_x,max_y = coordinates
        cv2.rectangle(boxed_image,(min_x,min_y),(max_x,max_y),(0,0,255),2)
    return(boxed_image)

#Checks for overlap of boxes showing same region

def non_max_suppression(boxes_ls, overlapThresh,debug):
    '''
    Uses method by Malisiewicz et al.
    Reads in array of boxes, if overlap>overlapThresh, merge boxes.
    '''
    # if there are no boxes, return an empty list
    if len(boxes_ls) == 0:
        return []

    boxes_arr = np.asarray(boxes_ls)

    # if the bounding boxes are integers, convert them to floats for divisions
    if boxes_arr.dtype.kind == "i":
        boxes_arr = boxes_arr.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes_arr[:,0]
    y1 = boxes_arr[:,1]
    x2 = boxes_arr[:,2]
    y2 = boxes_arr[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1) #+1 in case it is a point, prevents div 0 problem later
    idxs = np.argsort(y2)  #sort by ascending y2

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        #finding overlap area between current box and all other boxes by:
        # finding the largest NW (x, y) coordinates for the NW corner and the smallest (x, y) coordinates
        # for the SW corner - done by doing pairwise comparisons

        #compared coords are sorted by y2 (ascending) using the index generated earlier
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the overlapping box
        # if boxes do not overlap, w or h or both will be zero
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have areas ABOVE the threshold, including
        #current index being compared
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the integer data type
    return boxes_arr[pick].astype("int")


#Two possible implementations of final selection of merged boxes - either by area rank or by area threshold

def shortlist_by_rank(box_ls,rank_thresh):
    '''
    shortlists final selection of boxes by ranking by area, returns list of selected boxes
    '''
    box_area = []
    for box in new_boxes_supp:
        width = box[2]-box[0]
        height = box[3]-box[1]
        area = width*height
        box_area.append(area)

    ranked_by_area = sorted(list(zip(box_area,new_boxes_supp)),reverse = True)
    final_boxes = []
    print(ranked_by_area)

    for i in range(rank_thresh):
        box = list(ranked_by_area[i][1])
        print(box)
        final_boxes.append(box)

    return(final_boxes)

def shortlist_by_area(box_ls,area_thresh):
    '''
    shortlists final selection of boxes by defined area threshold
    '''
    final_boxes = []
    for box in box_ls:
        width = box[2]-box[0]
        height = box[3]-box[1]
        area = width*height
        if area>area_thresh:
            final_boxes.append(list(box))
    return final_boxes


#Utility function to generate list of image and coordinate directories for looping over

def get_im_coord_pair(pdf_file_name,im_folder,coord_folder):
    '''
    Generates list contaning pairs of image and coordinates file directories generated
    from splitting pdf. Used for looping over all the images and their respective coordiantes.
    '''
    image_file_dir = os.path.join(pdf_file_name,im_folder)
    image_file_ls = os.listdir(path=image_file_dir)

    coord_file_dir = os.path.join(pdf_file_name,coord_folder)
    coord_file_ls = os.listdir(path=coord_file_dir)

    return list(zip(image_file_ls,coord_file_ls))

def set_area_thresh(image,param,debug):
    '''
    Returns area threshold based on image properties.
    '''
    height,width = image.shape[0:2]
    area = height*width
    area_thresh = area/param
    if debug is True:
        print(f'Area threshold set as {area_thresh}')
    return area_thresh

def set_dist_thresh(image,param,debug):
    '''
    Returns distance threshold based on image properties
    '''
    height,width = image.shape[0:2]
    dist_thresh = height/param
    if debug is True:
        print(f'Distance threshold set as {dist_thresh}')
    return dist_thresh

def select_boxes(im_folder,coord_folder,crop_boxes_folder,pdf_file_name,im_coord_pair,dist_param,area_param,overlap_thresh):
    '''
    Takes in page_num+box_coordinates pair,dist and area parameters. Returns boxes in list for cropping.
    dist thresh = height/dist_param - increase dist param for more boxes
    area thresh = image area/area_param - increase area param for more boxes
    overlap thresh: minimum overlap which calls for merging
    '''

    boxes_ls = []

    for im_name, coord_name in im_coord_pair:
        #defining directories for accessing files
        im_dir = os.path.join(pdf_file_name,im_folder,im_name)
        image = cv2.imread(im_dir)

        #setting thresholds for deciding box merge and box selection
        dist_thresh = set_dist_thresh(image,dist_param,debug)
        area_thresh = set_area_thresh(image,area_param,debug)
        coord_dir = os.path.join(pdf_file_name,coord_folder,coord_name)
        minmax_list = readfile(coord_dir)

        cen_list = gen_cen_list(minmax_list)
        new_boxes_ls = []
        cen_minmax_dict = dict(zip(cen_list,minmax_list))


        #This merges nearby boxes (does not care about overlaps, checked by non_max_suppression) based
        #on distance
        #Each box only merged once - once boxes merged, remove from list of boxes being looped over
        while len(cen_list)>0:
            last = len(cen_list)-1  #takes centroid from the last of the index
            centroid = cen_list[last]

            #generate list of distance with all other centroids with those below threshold filtered out
            centroid_shortls,distance_shortls = centroid_shortlist(centroid, cen_list, dist_thresh)
            #merge starting from nearest, creating new box
            merged_box = merge_boxes(centroid,distance_shortls,cen_minmax_dict)
            #adds centroid being evaluated to list of merged centroids for it to be removed from looped list
            centroid_shortls.append(centroid)
            #remove merged centroids from centroid list
            cen_list = [x for x in cen_list if x not in centroid_shortls]
            #add newly merged box to list of merged boxes
            new_boxes_ls.append(merged_box)

        #Apply non-min max suppression to eliminate overlapping boxes
        new_boxes_ls = non_max_suppression(new_boxes_ls,overlap_thresh,debug)

        #Choose between selection using area or rank (comment out the method not chosen)
        #new_boxes_ls = shortlist_by_area(new_boxes_ls,area_thresh)
        #new_boxes_ls = shortlist_by_rank(new_boxes_ls,rank_thresh)

        boxes_ls.append((im_name,new_boxes_ls))

    return boxes_ls

#Reads in image and list of boxes to be cropped out
#Returns list of cropped images
def crop_image(image,boxes_ls):
    '''
    Crops out from an image using the list of box coordinates, returns list of cropped images
    '''
    cropped_images = []
    for coordinates in boxes_ls:
        x1,y1,x2,y2 = coordinates
        cropped_images.append(image[y1:y2,x1:x2])
    return cropped_images

def find_text_lines(pdf_file_name,output_folder,image,th):
    '''
    Finds the text lines- image array is reduced to vector
    and the average value is taken for each y coordinate.
    th: threshold for finding the transition from text to background
    '''

    #Checks that the image is grayscale, if not convert to grayscale
    if len(image.shape)>2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    text_lines = []

    #otsu thresholding  - another way of doing thresholding, tends to work less well
    #ret,thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #binary threshold
    ret,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

    #inverting image
    inverted_image = cv2.bitwise_not(thresh)
    hist = cv2.reduce(inverted_image,1,cv2.REDUCE_AVG).reshape(-1)
    H,W = image.shape
    lines = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th] #checking transition from black background to white
    lines.extend([0,H]) #adding lines to top and bottom of image
    lines.sort()

    #generating the two lines bound by combining adjacent top-bottom lines
    max_index = len(lines) - 1

    for i in range(max_index):
        top = lines[i]
        bottom = lines[i+1]
        if top<bottom:
            text_lines.append((0,top,W,bottom))

    return text_lines,inverted_image,lines

#Excel functions
def write_to_excel(worksheet,isbn_ls):
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

#(another function to verify and flag out in excel so that user knows the text is recognised)
def get_like_isbn(string,isbn_check):
    '''
    Takes in tesseract output and returns ISBN-like string
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


if __name__ == '__main__':
    #switch on debugging here
    debug = False

    #Tess parameters
    # Include the below line, if you don't have tesseract executable in your PATH
    #pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
    #Configuration for tesseract
    tess_config = "-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -psm 7"

    #folder names
    im_folder = 'images'
    coord_folder = 'coordinates'
    output_folder = 'output'
    crop_boxes_folder = 'final boxes'

    #key parameters
    angle=0
    resolution=300
    debug = False
    overlap_thresh = 0.3 #lower bound for overlap, adjust upwards for fewer merges
    line_thresh = 3
    dist_param = 3 #decrease for more merges
    area_param = 3000 #increase to keep smaller boxes
    isbn_check = '978' #used for checking validity of isbn

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="Save intermediate files in folders for debugging.", action="store_true")
    parser.add_argument("pdf_file", type=str, help="Select a file to extract ISBN from. Key it in the following format: 'example.pdf'")
    parser.add_argument("angle", type=int, help="Input the angle of clockwise rotation required: e.g. 90")
    args = parser.parse_args()

    if args.debug:
        debug = True
        print("Debug mode turned on.")

    pdf_file = args.pdf_file
    pdf_file_name = os.path.splitext(pdf_file)[0]

    angle = args.angle

    #File conversion
    convert_rotate(pdf_file,im_folder,coord_folder,output_folder,debug,angle=angle)

    #CTPN
    # ctpn configurations
    cfg_from_file('ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        if debug is True:
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    im_names = glob.glob(os.path.join(pdf_file_name, im_folder, '*.png')) + \
               glob.glob(os.path.join(pdf_file_name, im_folder, '*.jpg'))

    for im_name in im_names:
        if debug is True:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Identifying: {:s}'.format(im_name)))
        ctpn(sess, net, im_name,pdf_file_name,coord_folder,debug)

    if debug is True:
        print('***********************CTPN successful***********************')

    im_coord_pair = get_im_coord_pair(pdf_file_name,im_folder,coord_folder)
    selected_boxes_ls = select_boxes(im_folder,coord_folder,crop_boxes_folder,pdf_file_name,im_coord_pair,dist_param,area_param,overlap_thresh)

    #draws out final selected boxes
    if debug is True:
        for page_file_name,boxes_ls in selected_boxes_ls:
            image = cv2.imread(os.path.join(pdf_file_name,im_folder,page_file_name))
            boxed_image = draw_box(image,boxes_ls)
            cv2.imwrite(os.path.join(pdf_file_name,crop_boxes_folder,page_file_name),boxed_image)

    workbook = xlsxwriter.Workbook(os.path.join(pdf_file_name,'results.xlsx'))

    for page_file_name,boxes_ls in selected_boxes_ls:
        page_text_ls = []
        page_name = os.path.splitext(page_file_name)[0]
        worksheet = workbook.add_worksheet(page_name)
        create_column_titles(worksheet)

        page_directory = os.path.join(pdf_file_name,im_folder,page_file_name)
        page_image = cv2.imread(page_directory)
        cropped_im_ls = crop_image(page_image,boxes_ls)
        #loops over all the cropped images within a single page
        if debug is True:
            crop_index = 0

        for cropped_im in cropped_im_ls:
            text_lines_ls,inverted_image,lines = find_text_lines(pdf_file_name,output_folder,cropped_im,line_thresh)

            if debug is True:
                color = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)
                H,W,__ = cropped_im.shape
                for y in lines:
                    cv2.line(color, (0,y), (W, y), (255,0,0), 1)
                cv2.imwrite(os.path.join(pdf_file_name,output_folder,page_name+'_crop_'+str(crop_index)+'.png'),color)
                line_index = 0
            #loops over all the lines for a single cropped section within a single page

            for line_coords in text_lines_ls:
                line_coords_ls = [line_coords]  #crop_image takes in a list so first convert coords to a list
                line_image = crop_image(cropped_im,line_coords_ls)[0] #take image out of list
                line_text = pytesseract.image_to_string(line_image,config=tess_config).rstrip()
                line_text = get_like_isbn(line_text,isbn_check)
                if line_text is None:
                    pass
                else:
                    page_text_ls.append(line_text)
                #insert isbn check here

                if debug is True:
                    cv2.imwrite(os.path.join(pdf_file_name,output_folder,page_name+'_crop_'+str(crop_index)+'_line_'+
                        str(line_index)+'.png'),line_image)
                    line_index += 1
                    crop_index += 1
        verified_isbn_ls = verify_isbn(page_text_ls)
        write_to_excel(worksheet,verified_isbn_ls)

        if debug is True:
            print(f'This is the page text: {page_text_ls}')

        print(f'{page_file_name} done!')
        print('*************************************************')
    workbook.close()
    print('All done!')
    print(datetime.datetime.now())
