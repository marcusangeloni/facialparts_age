#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus.angeloni@ic.unicamp.br>
# Rodrigo de Freitas Pereira <rodrigodefreitas12@gmail.com>
# Helio Pedrini <helio@ic.unicamp.br>
# Tue 15 Jan 2019 18:00:00

import apache_beam as beam
import argparse
import io
import os
import json
import tempfile
import tensorflow as tf
import csv
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors
from apache_beam.options.pipeline_options import PipelineOptions

import logging
logging.basicConfig()
logger = logging.getLogger('Generate-TF-Records')
logger.setLevel(logging.INFO)

class ReadImages(beam.DoFn):
    def __init__(self, image_folder, pof, negative_list):
        super(ReadImages, self).__init__()
        self.image_folder = image_folder
        self.pof = pof
        self.negative_list = negative_list

    def process(self, element):
        image_name = element['image_name']
        image_label = int(element['image_label'])
        
        if image_name.split('/')[-1] not in self.negative_list:
            image_bytes = {}
            image_bytes['image_name'] = image_name.encode('utf-8')
            for _pof in self.pof:
                image_path = os.path.join(self.image_folder, _pof, image_name)
                logger.info("IMAGE_PATH: {}".format(image_path))
                image_bytes[_pof] = file_io.FileIO(image_path, mode='rb').read()            
            image_bytes['label'] = image_label

            yield image_bytes

class TFExampleFromImageDoFn(beam.DoFn):
    def __init__(self, pof):        
        super(TFExampleFromImageDoFn, self).__init__()
        self.pof = pof

    def process(self, element):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))

        def _float_feature(value):
            return tf.train.Feature(float_list = tf.train.FloatList(value = value))
        
        feat_dict = {k: _bytes_feature([element[k]]) for k in self.pof}
        feat_dict['label'] = _float_feature([element['label']])

        example = tf.train.Example(
            features = tf.train.Features(feature = feat_dict))
        yield example

def run(pipeline_args, known_args):
    logger.info(known_args)

    # list Parts Of Face (pof) directories
    pof = [f for f in os.listdir(known_args.image_dir) if os.path.isdir(
        os.path.join(known_args.image_dir, f))]
    logger.info("Parts of Faces: {}".format(pof))

    resize_config = json.load(open(known_args.resize_config_file, 'r'))
    logger.info('Resize config: {}'.format(resize_config))

    # split protocol-file to obtain task (age) and phase (train,test, val)
    task, phase = os.path.basename(
        known_args.protocol_file).split('.')[0].split('_')
    logger.info('Task: {}, Phase: {}'.format(task, phase))

    fold_id = os.path.abspath(known_args.protocol_file).split(
        '/')[-2].split('_')[-1]
    logger.info('Folder_id: {}'.format(fold_id))

    output_tfrecord = '{}/fold_{}_task_{}_phase_{}'.format(
        known_args.output_path, fold_id, task, phase)

    pipeline_options = PipelineOptions(pipeline_args)
    read_input_list = beam.io.ReadFromText(
        known_args.protocol_file, strip_trailing_newlines=True)

    if known_args.negative_list is not None:
        negative_list = [l.strip().split('/')[-1] for l in  open(known_args.negative_list,'r').readlines()]
    else:
        negative_list = None

    with beam.Pipeline(options=pipeline_options) as p:
        init = p | 'Start pipeline' >> beam.Create([''])
        (init |
            'read_image_list' >> read_input_list |
            'parse_list' >> beam.Map(lambda l: {'image_name': l.split(' ')[0], 'image_label': l.split(' ')[1]}) |
            'read_images' >> beam.ParDo(ReadImages(known_args.image_dir, pof,negative_list)) |
            'build_tfexample' >> beam.ParDo(TFExampleFromImageDoFn(pof)) |
            'serialize_tfexample' >> beam.Map(lambda x: x.SerializeToString()) |
            'write_tfrecord' >> beam.io.WriteToTFRecord(output_tfrecord, file_name_suffix='.tfrecord.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-file', dest = 'protocol_file', required = True)
    parser.add_argument('--image-dir', dest = 'image_dir', required = True)
    parser.add_argument('--output-path', dest = 'output_path', required = True)
    parser.add_argument('--resize-config-file',
                        dest = 'resize_config_file', required = True)
    parser.add_argument('--negative-list',
                        dest='negative_list', required = False,
                        default = None)
    known_args, pipeline_args = parser.parse_known_args()
    run(pipeline_args = pipeline_args, known_args=known_args)
