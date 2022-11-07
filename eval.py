#Required libraries

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm.notebook import tqdm as tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from getdata import get_data
from modelarch import create_model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
import argparse
import timeit

def loss_IoU(y_true, y_pred):    
    intersection_tensor=tf.math.multiply(y_true,y_pred)
    inter=tf.reduce_sum(intersection_tensor)    
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred),intersection_tensor))
    iou= tf.math.divide(inter,union)
    return 1-iou

#metric
def class_IoU(y_true, y_pred):    
    threshold = tf.constant([0.9])
    y_pred_threshold=tf.cast(tf.math.greater(y_pred, threshold),tf.int32)
    y_true=tf.cast(y_true,tf.int32)  
    intersection_tensor=tf.math.multiply(y_true,y_pred_threshold)
    inter=tf.reduce_sum(intersection_tensor)
    #union= a+b-intersection
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred_threshold),intersection_tensor))
    return tf.math.divide(inter,union)

def normalize(X):
    return (X / 255)

def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation on individual images')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input_folder', dest='input_folder', default='./dataset/input_images/aghi/', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='./bin/mobileNetv3_segmentation_new1.h5', type=str, help='path to the model to use')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #select the working GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[2], True)
    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    # cwd = os.getcwd()
    # print(cwd)
    # model_path = os.path.join(cwd,'bin','mobileNetv3_segmentation_new1.h5')
    model=tf.keras.models.load_model(args.model_path,
                                 custom_objects={"hard_swish": hard_swish,"loss_IoU": loss_IoU,"class_IoU":class_IoU}
                                )

    print('evaluating...')
    in_net_h=480#224
    in_net_w=640#224
    if args.input_folder.endswith('.png') or args.input_folder.endswith('.jpg'):

        x_test_new = cv2.imread(args.input_folder) 
        # x_test_new = X[0]
        x_test_new = cv2.cvtColor(x_test_new, cv2.COLOR_BGR2RGB)
        x_test_new = cv2.resize(x_test_new.astype('uint8')
                                , (in_net_h,in_net_w), interpolation = cv2.INTER_AREA)
        x_test_new = normalize(x_test_new)
        start = timeit.default_timer()
        y_pred = model.predict(x_test_new[None,...])
        stop = timeit.default_timer()
        y_pred = (y_pred > 0.5)
        dirname, basename = os.path.split(args.input_folder)
        save_path=args.pred_folder+basename[:-4]
        outimage = np.moveaxis(y_pred,0,-1)*255
        cv2.imwrite(save_path +"_pred_own"+'.jpg', outimage)
        print('Predicting the image took %f seconds'% (stop-start))
    else:
        dlist=os.listdir(args.input_folder)
        dlist.sort()
        time_sum = 0
        counter = 0
        for filename in dlist:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                path=args.input_folder+filename
                print("Predicting for:"+filename)
                img = cv2.imread(path)
                x_test_new = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2RGB)
                x_test_new = normalize(x_test_new)
                if counter==0:
                    start = timeit.default_timer()
                    y_pred = model.predict(x_test_new[None,...])
                    stop = timeit.default_timer()
                    setuptime = stop-start
                start = timeit.default_timer()
                y_pred = model.predict(x_test_new[None,...])
                stop = timeit.default_timer()
                if counter==0:
                    time_sum=stop-start
                    wsetuptime=setuptime
                else:
                    time_sum+=stop-start
                    wsetuptime+=stop-start
                threshold = np.mean(y_pred)
                y_pred[y_pred>=threshold]=255
                y_pred[y_pred<threshold]=0
                y_pred=y_pred.astype(np.uint8)
                y_pred= y_pred[0]
                y_pred3 = np.repeat(y_pred, 3, axis=-1)
                y_pred3_bool = (y_pred3>125)
                img = np.where(y_pred3_bool, img, img/3).astype(np.uint8)                
                counter=counter+1
                save_path=args.pred_folder+filename[:-4]
                cv2.imwrite(save_path +"_pred_deepseg_mask"+'.jpg', y_pred3)
                cv2.imwrite(save_path +"_pred_deepseg"+'.jpg', img)
            else:
                continue
        print('Predicting %d images took %f seconds, with the average of %f (including setup time: total: %fs, avg: %fs)' % (counter,time_sum,time_sum/counter,wsetuptime,wsetuptime/counter))  
    