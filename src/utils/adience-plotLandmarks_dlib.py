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
import dlib

img_extension = "jpg"

def find_biggest_face(dets):
    face = dlib.rectangle(0, 0, 0, 0)

    for k, d in enumerate(dets):
        if d.area() >= face.area():
            face = d

    return face

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description = 'Plot DLib facial landmarks in the Adience database')
parser.add_argument('predictor_path', default = '', help = 'Path to DLib shape predictor')
parser.add_argument('image_dir', default = '', help = 'Image directory')
parser.add_argument('output_dir', default = '', help = 'Output directory')

args = parser.parse_args()

if (not(os.path.exists(args.predictor_path))):
    print('DLib predictor path (\"' + args.predictor_path + '\") not found.')
    exit()

if (not(os.path.exists(args.image_dir))):
    print('Input image directory (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

predictor_path = args.predictor_path
image_dir = args.image_dir
output_dir = args.output_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - plot DLib landmarks started")
print("DLib predictor path: " + predictor_path)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# Initialize the face detector and shape predictor from DLib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# get the list of user directories
userList = os.listdir(image_dir)

for user in userList:
    user_path = os.path.join(image_dir, user) # define the user image path
    output_usr_path = os.path.join(output_dir, user) # define the output user path

    if (not(os.path.exists(output_usr_path))):
        os.mkdir(output_usr_path)

    imgList = os.listdir(user_path)

    for img in imgList:
        if(img[len(img)-3:len(img)] != img_extension): # check the file extension
            continue

        image_path = os.path.join(user_path, img) # define the input image path
        output_path = os.path.join(output_usr_path, img) # define the output image path

        # load the input image
        image = cv2.imread(image_path)
        print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + output_path)

        if os.path.exists(output_path):
            continue

        # Detect face
        dets = detector(image, 1)

        if (len(dets) <= 0):
            print("face not found " + str(len(dets)))
            d = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
        else:
            d = find_biggest_face(dets)

        shape = predictor(image, d)

        dets = detector(image, 1)

        # select a color for each facial part

        # 1-17 face contour
        color = (255, 0, 0) # blue
        for i in range(0, 17):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, -1)

        # 18-27 eyebrows
        color = (255, 255, 0) # cyan
        for i in range(17, 27):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, -1)

        # 28-36 nose
        color = (0, 0, 255) # red
        for i in range(27, 36):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, -1)

        # 37-48 eyes
        color = (0, 255, 0) # green
        for i in range(36, 48):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, -1)

        # 49-68 mouth
        color = (0, 255, 255) # yellow
        for i in range(48, 68):
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, color, -1)

        cv2.imwrite(output_path, image) # save the output image

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - plot DLib landmarks finished")
