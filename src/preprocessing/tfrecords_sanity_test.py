#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Tue 15 Jan 2019 18:00:00

import tensorflow as tf
import argparse
import json
import os
import numpy as np
import cv2
import base64

from google.protobuf.json_format import MessageToJson
from tensorflow.python.lib.io import file_io

import logging
logging.basicConfig()
logger = logging.getLogger('Sanity-TF-Records')
logger.setLevel(logging.INFO)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecord-file', dest = 'tfrecord_file', required=True)    
    parser.add_argument('--output-dir', dest = 'output_dir', required=True)    
    args = parser.parse_args()

    tf_iter = tf.python_io.tf_record_iterator(args.tfrecord_file, options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
    example = next(tf_iter)
    tf_example = tf.train.Example.FromString(example)
    jsonMessage = json.loads(MessageToJson(tf.train.Example.FromString(example)))

    logger.info(jsonMessage['features']['feature'].keys())

    for pof in jsonMessage['features']['feature']:
        if pof != "label" and pof != "image_name":
            with file_io.FileIO('{}.jpg'.format(os.path.join(args.output_dir, pof)),'wb') as f:
                img_str = base64.b64decode(jsonMessage['features']['feature'][pof]['bytesList']['value'][0])                
                f.write(img_str)
                nparr = np.fromstring(img_str, np.uint8)                
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
