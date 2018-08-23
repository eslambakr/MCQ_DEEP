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
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import time


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
#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

###############################
#   Predict bounding boxes
###############################
counter = 0
template = 100
num_stop = 5
if template == 20:
    num_of_question = 40
    num_of_choices = 4
    num_of_colums = 2
    num = 14
    start = 60
if template == 30:
    num_of_question = 50
    num_of_choices = 4
    num_of_colums = 2
    num = 6
    start = 77
if template == 40:
    num_of_question = 60
    num_of_choices = 4
    num_of_colums = 3
    num = 5
    start = 83
if template == 70:
    num_of_question = 90
    num_of_choices = 4
    num_of_colums = 3
    num = 12
    start = 88
if template == 100:
    num_of_question = 120
    num_of_choices = 4
    num_of_colums = 4
    num = 11
    start = 90
    num_stop = 8

for i in tqdm(range(1)):
    s_total = time.clock()
    start=0
    counter = i+start
    x = str(counter).zfill(4)
    s_time1 = time.clock()
    #image_path = "/home/eslam/PycharmProjects/yolo_keras/venv/keras-yolo2-master/images/mcq_test_circul/"+x+".jpg"
    image_path = "/home/eslam/PycharmProjects/yolo_keras/venv/keras-yolo2-master/images/test_temp_100/" + x + ".jpg"
    image_path = "/home/eslam/0004.jpg"
    image_path = "/home/eslam/PycharmProjects/yolo_keras/venv/keras-yolo2-master/images/test_temp_100/0111.jpg"
    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    e_time1 = time.clock()
    print(len(boxes), 'boxes are found')
    s_time2 = time.clock()
    image_h, image_w, _ = image.shape
    center_x = np.zeros(num_of_question*num_of_choices+5)
    center_y = np.zeros(num_of_question*num_of_choices+5)
    labels = np.zeros(num_of_question * num_of_choices+5)
    x = np.zeros(num_stop)
    y = np.zeros(num_stop)
    x_first3 = np.zeros(8)
    y_first3 = np.zeros(8)
    error = np.zeros(num_of_choices * num_of_question)
    error_index = np.zeros(num_of_choices * num_of_question - num_of_choices)
    boxes_in_lines = np.zeros((int(num_of_question / num_of_colums),num_of_choices*num_of_colums,4))
    eslam = np.zeros((int(num_of_question / num_of_colums), num_of_choices * num_of_colums, 4))
    i=0

    if len(boxes) != 485:
        print("Wrong Detection !!!!!!!!!!!!!!!!")
    count_first_5 = 0
    center_x_first5 = np.zeros(5)
    center_y_first5 = np.zeros(5)
    labels_first5 = np.zeros(5)
    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)
        center_x[i] = int(.5 * (xmin+ xmax))
        center_y[i] = int(.5 * (ymin + ymax))
        labels[i] = int(box.get_label())
        #cv2.rectangle(image, (int(center_x_y[i][0]), int(center_x_y[i][1])), (int(center_x_y[i][0])+10, int(center_x_y[i][1])+10), (0, 255, 0), 3)
        i=i+1
    #cv2.imwrite(image_path[:-4] + '_detected_' + image_path[-4:], image)
    center_x_y = np.array(list(zip(center_x,center_y,labels,error)), dtype=[('x', float), ('y', float), ('l', int),('e',float)])
    sorted_center_x_y = np.sort(center_x_y,order="y")
    best_y = sorted_center_x_y[:2]
    x_y_counter = 1
    for counter in range(8):
        x_first3[counter] = sorted_center_x_y[counter][0]
        y_first3[counter] = sorted_center_x_y[counter][1]
    s_time3 = time.clock()
    print("the first 8 points is = ",x_first3)
    coeffs_first3 = poly.polyfit(x_first3, y_first3, 1)
    e_time3 = time.clock()
    """
    for index in range(8):
        cv2.rectangle(image, (int(x_first3[index]), int(y_first3[index])),
                  (int(x_first3[index]) + 10, int(y_first3[index]) + 10), (255, 255, 0), 3)
    """
    print(coeffs_first3)
    for index in range(num_of_choices*num_of_colums*3):
        if index ==0:
            x[0]= sorted_center_x_y[index][0]
            x_min = x[0]
            y[0] = sorted_center_x_y[index][1]
        else:
            if coeffs_first3[1] > -3e-02 and coeffs_first3[1] < 3e-02:
                x[x_y_counter] = sorted_center_x_y[index][0]
                y[x_y_counter] = sorted_center_x_y[index][1]
                x_y_counter = x_y_counter + 1
                if x_y_counter == num_stop:
                    break
            elif coeffs_first3[1] > 3e-02: # rotation right
                if sorted_center_x_y[index][0] > x_min:
                    x[x_y_counter] = sorted_center_x_y[index][0]
                    x_min = x[x_y_counter]
                    y[x_y_counter] = sorted_center_x_y[index][1]
                    x_y_counter = x_y_counter + 1
                    if x_y_counter ==num_stop:
                        break
            elif coeffs_first3[1] < -3e-02: # rotation left
                if sorted_center_x_y[index][0] < x_min:
                    x[x_y_counter] = sorted_center_x_y[index][0]
                    x_min = x[x_y_counter]
                    y[x_y_counter] = sorted_center_x_y[index][1]
                    x_y_counter = x_y_counter + 1
                    if x_y_counter ==num_stop:
                        break
    """
    for index in range(5):
        cv2.rectangle(image, (int(x[index]), int(y[index])),
                  (int(x[index]) + 10, int(y[index]) + 10), (0, 255, 0), 3)
    """
    x_axies = np.linspace(0, 1280, 10)
    # Get the best line fit the points
    coeffs = poly.polyfit(x, y,1)
    z = poly.polyval(x_axies,coeffs)
    # Get the best points match the line
    for index in range(num_of_choices*num_of_question):
        new_y = poly.polyval(sorted_center_x_y[index][0], coeffs)
        sorted_center_x_y[index][3] = abs(new_y-sorted_center_x_y[index][1])
    # Sort the array according to the error
    sorted_error = np.sort(sorted_center_x_y,order="e")
    # Split the boxes to lines
    for index in range(int(num_of_question / num_of_colums)):
        for box_num in range(num_of_colums*num_of_choices):
            boxes_in_lines[index][box_num][0] = sorted_error[(index * num_of_colums * num_of_choices) + box_num][0]
            boxes_in_lines[index][box_num][1] = sorted_error[(index * num_of_colums * num_of_choices) + box_num][1]
            boxes_in_lines[index][box_num][2] = sorted_error[(index * num_of_colums * num_of_choices) + box_num][2]
            boxes_in_lines[index][box_num][3] = sorted_error[(index * num_of_colums * num_of_choices) + box_num][3]
    #print(boxes_in_lines)
    ## for 100 Template onlyyyyyyyyyyyyyyy !
    if template == 100:
        x_for_100_only = np.zeros(16)
        y_for_100_only = np.zeros(16)
        points_for_100_only = np.array(list(zip(np.zeros(320), np.zeros(320), np.zeros(320), np.zeros(320))),
                              dtype=[('x', float), ('y', float), ('l', int), ('e', float)])
        for i in range(16):
            x_for_100_only[i] = boxes_in_lines[9][i][0]
            y_for_100_only[i] = boxes_in_lines[9][i][1]
        cof_for_100_only = poly.polyfit(x_for_100_only, y_for_100_only, 1)
        # Get the best points match the line
        for index in range(320):
            new_y = poly.polyval(sorted_error[index+160][0], cof_for_100_only)
            points_for_100_only[index][0] = sorted_error[index+160][0]
            points_for_100_only[index][1] = sorted_error[index+160][1]
            points_for_100_only[index][2] = sorted_error[index+160][2]
            points_for_100_only[index][3] = abs(new_y - sorted_error[index+160][1])
        # Sort the array according to the error
        points_for_100_only_err_sorted = np.sort(points_for_100_only, order="e")
        # Split the boxes to lines
        for index in range(20):
            for box_num in range(num_of_colums * num_of_choices):
                boxes_in_lines[index+10][box_num][0] = points_for_100_only_err_sorted[(index * num_of_colums * num_of_choices) + box_num][0]
                boxes_in_lines[index+10][box_num][1] = points_for_100_only_err_sorted[(index * num_of_colums * num_of_choices) + box_num][1]
                boxes_in_lines[index+10][box_num][2] = points_for_100_only_err_sorted[(index * num_of_colums * num_of_choices) + box_num][2]
                boxes_in_lines[index+10][box_num][3] = points_for_100_only_err_sorted[(index * num_of_colums * num_of_choices) + box_num][3]
     ## for 100 Template onlyyyyyyyyyyyyyyy !
        x_for_100_only = np.zeros(16)
        y_for_100_only = np.zeros(16)
        points_for_100_only2 = np.array(list(zip(np.zeros(160), np.zeros(160), np.zeros(160), np.zeros(160))),
                                       dtype=[('x', float), ('y', float), ('l', int), ('e', float)])
        for i in range(16):
            x_for_100_only[i] = boxes_in_lines[19][i][0]
            y_for_100_only[i] = boxes_in_lines[19][i][1]
        cof_for_100_only = poly.polyfit(x_for_100_only, y_for_100_only, 1)
        # Get the best points match the line
        for index in range(160):
            new_y = poly.polyval(points_for_100_only_err_sorted[index + 160][0], cof_for_100_only)
            points_for_100_only2[index][0] = points_for_100_only_err_sorted[index + 160][0]
            points_for_100_only2[index][1] = points_for_100_only_err_sorted[index + 160][1]
            points_for_100_only2[index][2] = points_for_100_only_err_sorted[index + 160][2]
            points_for_100_only2[index][3] = abs(new_y - points_for_100_only_err_sorted[index + 160][1])
        # Sort the array according to the error
        points_for_100_only_err_sorted2 = np.sort(points_for_100_only2, order="e")
        # Split the boxes to lines
        for index in range(10):
            for box_num in range(num_of_colums * num_of_choices):
                boxes_in_lines[index + 20][box_num][0] = \
                points_for_100_only_err_sorted2[(index * num_of_colums * num_of_choices) + box_num][0]
                boxes_in_lines[index + 20][box_num][1] = \
                points_for_100_only_err_sorted2[(index * num_of_colums * num_of_choices) + box_num][1]
                boxes_in_lines[index + 20][box_num][2] = \
                points_for_100_only_err_sorted2[(index * num_of_colums * num_of_choices) + box_num][2]
                boxes_in_lines[index + 20][box_num][3] = \
                points_for_100_only_err_sorted2[(index * num_of_colums * num_of_choices) + box_num][3]
    # Arrange the boxes in X axes
    i=0
    for line in boxes_in_lines:
        eslam[i] = sorted(line, key=lambda x: x[0])
        i= i+1
    #print(eslam)
    print(eslam[0][0][0])
    print(eslam[0][15][0])
    e_time2 = time.clock()
    e_total = time.clock()
    # Draw the line for debugging
    lineThickness = 2
    cv2.line(image, (int(x_axies[0]), int(z[0])), (int(x_axies[-1]), int(z[-1])), (0, 255, 0), lineThickness)
    # draw the lines
    for index in range(int(num_of_question / num_of_colums)):
        cv2.line(image, (int(eslam[index][0][0]), int(eslam[index][0][1])),
             (int(eslam[index][-1][0]), int(eslam[index][-1][1])), (200, 0, 0), lineThickness)

    cv2.imwrite(image_path[:-4] + '_detected_' + image_path[-4:], image)
    counter=counter+1

    print("the detection = ",((e_time1-s_time1)/(e_total-s_total))*100)
    print("2nd block = ",((e_time2-s_time2)/(e_total-s_total))*100)
    print("fit block = ", ((e_time3 - s_time3)/(e_total-s_total))*100)
