#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes,draw_boxes_only
from frontend import YOLO
import json
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



config_path = "/home/eslam/PycharmProjects/yolo_keras/venv/config.json"
weights_path = "/home/eslam/PycharmProjects/yolo_keras/venv/c58_.75.h5"

with open(config_path) as config_buffer:
    config = json.load(config_buffer)

###############################
#   Make the model
###############################

yolo = YOLO(backend=config['model']['backend'],
            input_size=config['model']['input_size'],
            labels=config['model']['labels'],
            max_box_per_image=config['model']['max_box_per_image'],
            anchors=config['model']['anchors'])

###############################
#   Load trained weights
###############################

yolo.load_weights(weights_path)

my_graph = tf.keras.backend.get_session().graph
latest_checkpoint ="/home/eslam"
tf.train.write_graph(my_graph, latest_checkpoint,'train.pbtxt',as_text=True )
tf.train.write_graph(my_graph, latest_checkpoint, 'train.pb', as_text=False)
print(yolo.model.output.op.name)#lambda_1/Identity
yolo.model.save("c_50*50_otpimaized.h5")
#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

###############################
#   Predict bounding boxes
###############################
begin = 0
for i in range(115):
    counter = begin + i
    x = str(counter).zfill(4)
    #image_path = "/home/eslam/PycharmProjects/yolo_keras/venv/keras-yolo2-master/images/test_temp_100/"+x+".jpg"
    image_path = "/home/eslam/2101.jpg"
    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    #image = draw_boxes(image, boxes, config['model']['labels'])
    image = draw_boxes_only(image, boxes)

    print(len(boxes), 'boxes are found  ',x)

    cv2.imwrite(image_path[:-4] + '_detected_' + image_path[-4:], image)
    #counter=counter+1