#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Wed 16 Dec 2018 21:00:00

import os
import cv2
import numpy
import argparse
from datetime import datetime
import pandas

img_extension = "jpg"

def read_openface_landmarks(coord_file):
    df = pandas.read_csv(coord_file, header = 0)
    idx = 0

    if df.shape[0] > 1:
        print("rows = " + str(df.shape[0]))

        min_distance = 10000 
        for i in range(0, df.shape[0]):
            mean_x = (df.values[i, 32] + df.values[i, 35]) / 2 # nose point 30 and 33
            mean_y =  (df.values[i, 100] + df.values[i, 103]) / 2 # nose point 30 and 33

            distance = numpy.sqrt(numpy.square(mean_x - 408) + numpy.square(mean_y - 408)) # euclidean distance to image center
            if distance < min_distance:
                idx = i
                min_distance = distance

    landmarks = df.values[idx, 2:]
    return landmarks        
    # image of aligned dir has 816x816 pixels (center is on 408,408)

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description = 'Plot OpenFace facial landmarks in the Adience database')
parser.add_argument('image_dir', default = '', help = 'Image directory')
parser.add_argument('coords_dir', default = '', help = 'Landmarks directory')
parser.add_argument('output_dir', default = '', help = 'Output directory')

args = parser.parse_args()

if (not(os.path.exists(args.coords_dir))):
    print('Landmarks directory (\"' + args.coords_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.image_dir))):
    print('Input image directory (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

coords_dir = args.coords_dir
image_dir = args.image_dir
output_dir = args.output_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - plot OpenFace landmarks started")
print("OpenFace landmarks directory: " + coords_dir)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# get the list of user directories
userList = os.listdir(image_dir)

for user in userList:
    user_path = os.path.join(image_dir, user) # get the user image path
    coords_usr_path = os.path.join(coords_dir, user) # get the user coordinates path
    output_usr_path = os.path.join(output_dir, user) # define the output user path
    
    if (not(os.path.exists(output_usr_path))):
        os.mkdir(output_usr_path)

    imgList = os.listdir(user_path)

    for img in imgList:
        if(img[len(img) - 3:len(img)] != img_extension): # check the file extension
            continue

        image_path = os.path.join(user_path, img) # get the input image path
        output_path = os.path.join(output_usr_path, img) # define the output image path
        coords_path = os.path.join(coords_usr_path, img.replace(".jpg", ".csv"))

        # load the input image
        image = cv2.imread(image_path)
        print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + output_path)

        if os.path.exists(output_path):
            continue

        # Check if landmarks exists (if face was found)
        if not(os.path.exists(coords_path)):
            print("face not found")
            continue

        landmarks = read_openface_landmarks(coords_path)

        # select a color for each facial part

        # 1-17 face contour
        color = (255, 0, 0) # blue
        for i in range(0, 17):
            cv2.circle(image, (int(landmarks[i]), int(landmarks[i + 68])), 1, color, -1)

        # 18-27 eyebrows
        color = (255, 255, 0) # cyan
        for i in range(17, 27):
            cv2.circle(image, (int(landmarks[i]), int(landmarks[i + 68])), 1, color, -1)

        # 28-36 nose
        color = (0, 0, 255) # red
        for i in range(27, 36):
            cv2.circle(image, (int(landmarks[i]), int(landmarks[i + 68])), 1, color, -1)

        # 37-48 eyes
        color = (0, 255, 0) # green
        for i in range(36, 48):
            cv2.circle(image, (int(landmarks[i]), int(landmarks[i + 68])), 1, color, -1)

        # 49-68 mouth
        color = (0, 255, 255) # yellow
        for i in range(48, 68):
            cv2.circle(image, (int(landmarks[i]), int(landmarks[i + 68])), 1, color, -1)

        cv2.imwrite(output_path, image) # save the output image

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - plot Openface landmarks finished")
