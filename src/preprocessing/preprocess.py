#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Mon 7 Jan 2019 18:00:00

import sys
import scipy.io
import os
import cv2
import numpy
import math
import scipy.ndimage

# get the coordinates angle to image alignment
def get_angle(x1, y1, x2, y2):
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle

# based on the angle of the coordinates segment the image and landmark coordinates are rotated
def rotate_image_and_coordinates(image, coord, angle):
    angleInDegrees = angle * 180 / math.pi
    centerY = image.shape[0] / 2
    centerX = image.shape[1] / 2

    outputImage = scipy.ndimage.rotate(image, angleInDegrees, (1, 0), False)
    outputCoordinates = numpy.zeros(coord.shape)

    for idx in range(0, len(coord)):
        if (coord[idx, 0] > 0) and (coord[idx, 1] > 0):
            outputCoordinates[idx, 0] = math.cos(angle) * (coord[idx, 0] - centerX) + math.sin(angle) * (coord[idx, 1] - centerY) + centerX
            outputCoordinates[idx, 1] = -math.sin(angle) * (coord[idx, 0] - centerX) + math.cos(angle) * (coord[idx, 1] - centerY) + centerY

    return(outputImage, outputCoordinates)

# find the maximum and minimum landmark coordinates of each facial part
def find_min_max(coordinates):
    size = coordinates.shape

    maxX = coordinates[0, 0]
    maxY = coordinates[0, 1]
    
    if maxX > 0:
        minX = maxX
    else:
        minX = 1000
    
    if maxY > 0:
        minY = maxY
    else:
        minY = 1000

    for idx in range(1, size[0]):
        if (coordinates[idx][0] < minX) and (coordinates[idx][0] > 0):
            minX = coordinates[idx][0]
        elif coordinates[idx][0] > maxX:
            maxX = coordinates[idx][0]

        if (coordinates[idx][1] < minY) and (coordinates[idx][1] > 0):
            minY = coordinates[idx][1]
        elif coordinates[idx][1] > maxY:
            maxY = coordinates[idx][1]

    return (minX, maxX, minY, maxY)

# crop the roi of each facial part and resize
def crop_and_resize(image, minX, maxX, minY, maxY, incr, width, height):
    increase = (maxX - minX) * incr # incr % in horizontal direction
    minX = minX - increase
    maxX = maxX + increase
    
    w = maxX - minX
    h = maxY - minY
    centerY = minY + h / 2

    newHeight = w * height / width # crop size
    minY = centerY - newHeight / 2
    maxY = centerY + newHeight / 2

    # test the limits of the image before crop it
    if (minX < 0):
        maxX -= minX
        minX = 0
    if (maxX > image.shape[1]):
        maxX = image.shape[1]
    if (minY < 0):
        maxY -= minY
        minY = 0
    if (maxY > image.shape[0]):
        maxY = image.shape[0]

    cropImg = image[int(minY):int(maxY), int(minX):int(maxX), 0:3]
    cropImg = cv2.resize(cropImg, (width, height))

    return cropImg

# load image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# create the facial parts directories to save facial parts in numpy format
def create_facial_parts_dir(output_dir):
    eyebrows_dir = os.path.join(output_dir, "eyebrows")
    eyes_dir = os.path.join(output_dir, "eyes")
    nose_dir = os.path.join(output_dir, "nose")
    mouth_dir = os.path.join(output_dir, "mouth")

    if not os.path.exists(eyebrows_dir):
        os.mkdir(eyebrows_dir)
    if not os.path.exists(eyes_dir):
        os.mkdir(eyes_dir)
    if not os.path.exists(nose_dir):
        os.mkdir(nose_dir)
    if not os.path.exists(mouth_dir):
        os.mkdir(mouth_dir)

    return (eyebrows_dir, eyes_dir, nose_dir, mouth_dir)

# convert from dLib to numpy array 
def shape_to_np(shape, dtype="float"):
    # initialize the list of (x, y)-coordinates
    coords = numpy.zeros((68, 2), dtype=dtype) 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords
