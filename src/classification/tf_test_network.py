#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Wed 6 Feb 2019 13:00:00

from __future__ import division

import tensorflow as tf
import os
import csv
import numpy as np
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

# read the list file from protocol and return the trials and respective ground truth
def list_images(list_file):
    trials = []
    ground_truth = []

    with open(list_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = " ")
        for row in reader:
            trials.append(row[0])
            ground_truth.append(int(row[1]))

    return trials, np.array(ground_truth)


#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description = 'Predict and compute metrics of a fold from ADIENCE Dataset')
parser.add_argument('facialparts_dir', default = '', help = 'Full path of facial parts images')
parser.add_argument('protocol_dir', default = '', help = 'Full path of protocol files')
parser.add_argument('model_path', default = '', help = 'Full path of CNN trained model')
parser.add_argument('fold', default='', help = 'Fold number [0-4]')


args = parser.parse_args()

if (not(os.path.exists(args.facialparts_dir))):
    print('Facial parts images (\"' + args.facialparts_dir + '\") not found.')
    exit()
    
if (not(os.path.exists(args.protocol_dir))):
    print('Protocol files (\"' + args.model_path + '\") not found.')
    exit()

if (not(os.path.exists(args.model_path))):
    print('Model (\"' + args.model_path + '\") not found.')
    exit()

if (int(args.fold) < 0 or int(args.fold) > 4):
    print('Fold (\"' + args.fold + '\") not supported.')
    exit()

model_path = args.model_path
fold = args.fold
protocol_dir = args.protocol_dir
facialparts_dir = args.facialparts_dir

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Prediction started")
print("Model path: " + model_path)
print("Fold: " + fold)

# list file with test trials
test_path = os.path.join(protocol_dir,"test_fold_is_" + str(fold), "age_test.txt")
trials, ground_truth = list_images(test_path)

# trained model
model = tf.contrib.predictor.from_saved_model(model_path)

pred = []
for trial in tqdm(trials):
    
    # load facial parts
    with open(os.path.join(facialparts_dir, "eyebrows", trial), 'rb') as f:
        eyebrows_bytes = f.read()
    with open(os.path.join(facialparts_dir, "eyes", trial), 'rb') as f:
        eyes_bytes = f.read()
    with open(os.path.join(facialparts_dir, "nose", trial), 'rb') as f:
        nose_bytes = f.read()
    with open(os.path.join(facialparts_dir, "mouth", trial), 'rb') as f:
        mouth_bytes = f.read()

    # inference
    predict = model({'eyebrows':[eyebrows_bytes], 'eyes':[eyes_bytes], 'nose':[nose_bytes], 'mouth':[mouth_bytes]})
    pred.append(predict['softmax'][0].argmax())

# metrics
pred = np.array(pred)
np.savetxt(os.path.join(protocol_dir, str(fold) + "predictions.txt"), pred)

correct = (pred == ground_truth)
correct_1off = (np.abs(pred - ground_truth) <= 1)
print("Total trials: " + str(correct.size))
print("Accuracy: " + str(float(correct.sum()) / correct.size))
print("Accuracy 1-off: " + str(float(correct_1off.sum()) / correct_1off.size))
