#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Wed 9 Jan 2019 20:00:00

import sys
import os
import numpy
import argparse
from datetime import datetime
import dlib
import preprocess
import cv2
import pandas

image_extension    = "jpg"

def find_biggest_face(dets):
    face = dlib.rectangle(0, 0, 0, 0)

    for k, d in enumerate(dets):
        if d.area() >= face.area():
            face = d

    return face

def read_openface_landmarks(coord_file):
    df = pandas.read_csv(coord_file, header=0)
    idx = 0

    if df.shape[0] > 1:
        min_distance = 10000 
        for i in range(0, df.shape[0]):
            mean_x = (df.values[i, 32] + df.values[i, 35]) / 2 # nose point 30 and 33
            mean_y =  (df.values[i, 100] + df.values[i, 103]) / 2 # nose point 30 and 33

            # image of aligned dir has 816x816 pixels (center is 408,408)
            distance = numpy.sqrt(numpy.square(mean_x - 408) + numpy.square(mean_y - 408)) # euclidean distance to image center
            if distance < min_distance:
                idx = i
                min_distance = distance

    landmarks = df.values[idx, 2:]
    
    # initialize the list of (x, y)-coordinates
    coords = numpy.zeros((68, 2), dtype = float)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (landmarks[i], landmarks[i + 68])
    
    return coords

def processed_before(eyebrows_usr_path, eyes_usr_path, nose_usr_path, mouth_usr_path, img):
    return (os.path.exists(os.path.join(eyebrows_usr_path, img)) and 
        os.path.exists(os.path.join(eyes_usr_path, img)) and
        os.path.exists(os.path.join(nose_usr_path, img)) and
        os.path.exists(os.path.join(mouth_usr_path, img)))

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description = 'Adience preprocessing')
parser.add_argument('predictor_path', default = '', help = 'Path to DLib shape predictor')
parser.add_argument('image_dir', default = '', help = 'Image directory')
parser.add_argument('coords_dir', default = '', help = 'OpenFace landmarks directory')
parser.add_argument('output_dir', default = '', help = 'Output directory')

args = parser.parse_args()

if (not(os.path.exists(args.coords_dir))):
    print('Coords directory (\"' + args.coords_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.predictor_path))):
    print('DLib predictor path (\"' + args.annotation_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.image_dir))):
    print('Input image directory (\"' + args.image_dir + '\") not found.')
    exit()

if (not(os.path.exists(args.output_dir))):
    os.mkdir(args.output_dir)

predictor_path = args.predictor_path
coords_dir = args.coords_dir
image_dir = args.image_dir
output_dir = args.output_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - preprocessing started")
print("OpenFace landmarks directory: " + coords_dir)
print("DLib predictor path: " + predictor_path)
print("Image directory: " + image_dir)
print("Output directory: " + output_dir)

# Initialize the face detector and shape predictor from DLib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# get the list of user directories
userList = os.listdir(image_dir)

eyebrows_dir, eyes_dir, nose_dir, mouth_dir = preprocess.create_facial_parts_dir(output_dir)

for user in userList:
    user_path = os.path.join(image_dir, user) # define the user directory path
    coords_usr_path = os.path.join(coords_dir, user)

    # define the output parts path
    eyebrows_usr_path = os.path.join(eyebrows_dir, user)
    eyes_usr_path = os.path.join(eyes_dir, user)
    nose_usr_path = os.path.join(nose_dir, user)
    mouth_usr_path = os.path.join(mouth_dir, user)

    if (not(os.path.exists(eyebrows_usr_path))):
        os.mkdir(eyebrows_usr_path)
    if (not(os.path.exists(eyes_usr_path))):
        os.mkdir(eyes_usr_path)
    if (not(os.path.exists(nose_usr_path))):
        os.mkdir(nose_usr_path)
    if (not(os.path.exists(mouth_usr_path))):
        os.mkdir(mouth_usr_path)

    imgList = os.listdir(user_path) # get the user's images

    for img in imgList:
        if(img[len(img) - 3:len(img)] != image_extension): # check the file extension
            continue

        image_path = os.path.join(user_path, img) # define the input image path
        print (datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Current image " + image_path)

        if processed_before(eyebrows_usr_path, eyes_usr_path, nose_usr_path, mouth_usr_path, img):
            continue

        # load the input image
        image = preprocess.load_image(image_path)
        coords_path = os.path.join(coords_usr_path, img.replace(".jpg", ".csv"))

        if not(os.path.exists(coords_path)): # if openface did not find landmarks
            print("openface did not find landmarks")
            
            # Detect face using Dlib
            dets = detector(image, 1)

            if (len(dets) <= 0):
                print("face not found " + str(len(dets)))
                d = dlib.rectangle(0,0,image.shape[1], image.shape[0])
            else:
                d = find_biggest_face(dets)

            shape = predictor(image, d)

            # Convert landmarks to numpy array
            coordinates = preprocess.shape_to_np(shape)
        else:
            coordinates = read_openface_landmarks(coords_path)
            
        #### eyebrows
        angle = preprocess.get_angle(coordinates[21, 0], coordinates[21, 1], coordinates[22, 0], coordinates[22, 1])
        eyebrows_img, coords = preprocess.rotate_image_and_coordinates(image, coordinates, angle)
        
        # segment the eyebrows region
        coords = coords[17:27, 0:2]
        minX, maxX, minY, maxY = preprocess.find_min_max(coords)

        try:
            eyebrows_img = preprocess.crop_and_resize(eyebrows_img, minX, maxX, minY, maxY, 0.03, 228, 33)
        except:
            eyebrows_img = cv2.resize(image, (228, 33))
            print("Eyebrows not found") # negative list        

        #### eyes
        angle = preprocess.get_angle(coordinates[39, 0], coordinates[39, 1], coordinates[42, 0], coordinates[42, 1])
        eyes_img, coords = preprocess.rotate_image_and_coordinates(image, coordinates, angle)
        
        # segment the eyes region
        coords = coords[36:48, 0:2]
        minX, maxX, minY, maxY = preprocess.find_min_max(coords)

        try:
            eyes_img = preprocess.crop_and_resize(eyes_img, minX, maxX, minY, maxY, 0.03, 202, 38)
        except:
            eyes_img = cv2.resize(image, (202, 38))
            print("Eyes not found") # negative list             

        #### nose
        angle = preprocess.get_angle(coordinates[32, 0], coordinates[32, 1], coordinates[34, 0], coordinates[34, 1])
        nose_img, coords = preprocess.rotate_image_and_coordinates(image, coordinates, angle)
        
        # segment the nose region
        coords = coords[27:36, 0:2]
        minX, maxX, minY, maxY = preprocess.find_min_max(coords)

        try:
            nose_img = preprocess.crop_and_resize(nose_img, minX, maxX, minY, maxY, 0.4, 103, 73)
        except:
            nose_img = cv2.resize(image, (103, 73))
            print("Nose not found") # negative list

        #### mouth
        angle = preprocess.get_angle(coordinates[48, 0], coordinates[48, 1], coordinates[54, 0], coordinates[54, 1])
        mouth_img, coords = preprocess.rotate_image_and_coordinates(image, coordinates, angle)
        
        # segment the mouth region
        coords = coords[48:68, 0:2]
        minX, maxX, minY, maxY = preprocess.find_min_max(coords)

        try:
            mouth_img = preprocess.crop_and_resize(mouth_img, minX, maxX, minY, maxY, 0.08, 114, 66)
        except:
            mouth_img = cv2.resize(image, (114, 66))
            print("Mouth not found") # negative list

        # save the facial parts images
        if not(eyebrows_img is None):
            cv2.imwrite(os.path.join(eyebrows_usr_path, img), eyebrows_img)
        if not(eyes_img is None):
            cv2.imwrite(os.path.join(eyes_usr_path, img), eyes_img)
        if not(nose_img is None):
            cv2.imwrite(os.path.join(nose_usr_path, img), nose_img)
        if not(mouth_img is None):
            cv2.imwrite(os.path.join(mouth_usr_path, img), mouth_img)
            
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Adience - preprocessing finished")
