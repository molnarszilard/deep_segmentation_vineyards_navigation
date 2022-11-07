#Required libraries
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from getdata import get_data
from modelarch import create_model
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s --- %s/%s %s\r' % (bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

def loss_IoU(y_true, y_pred):  
    intersection_tensor=tf.math.multiply(y_true,y_pred)
    inter=tf.reduce_sum(intersection_tensor)    
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred),intersection_tensor))
    iou= tf.math.divide(inter,union)
    return 1-iou

#metric
def class_IoU(y_true, y_pred): 
    # threshold = tf.constant([0.5])
    threshold = tf.reduce_mean(y_pred)
    y_pred_threshold=tf.cast(tf.math.greater(y_pred, threshold),tf.int32)
    y_true=tf.cast(y_true,tf.int32)  
    intersection_tensor=tf.math.multiply(y_true,y_pred_threshold)
    inter=tf.reduce_sum(intersection_tensor)
    #union= a+b-intersection
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred_threshold),intersection_tensor))
    return tf.math.divide(inter,union)

#select the working GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
tf.config.experimental.set_memory_growth(gpus[2], True)

### DATASET
PATH_DIR = 'dataset/aghi_mod/'
in_net_h=480
in_net_w=640
#or
# in_net_h=224
# in_net_w=224
net_channels=3
n_epochs = 100
X,y,X_test,y_test = get_data(PATH_DIR,in_net_h,in_net_w)

#Save checkpoints
model_dir = './bin/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)    
name = 'MobileNet_V3'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
logdir = "{}/run-{}_{}/".format(root_logdir, now,name)
backup_model_path = os.path.join(model_dir, '{}.h5'.format(name))
backup_weights_path = os.path.join(model_dir, '{}_weights.h5'.format(name))
checkpointer = ModelCheckpoint(filepath=backup_weights_path, 
                               monitor = 'loss',
                               verbose=1, 
                               save_best_only=True)
model = create_model(in_net_w,in_net_h)
#Define the optimizers
optimizer_r = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer_a = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
optimizer_s=SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
#Compile the model 
model.compile(optimizer=optimizer_s, loss=loss_IoU, metrics = [class_IoU])
#The first layers have been already frozen previously
early_stopping_ = tf.keras.callbacks.EarlyStopping(patience=n_epochs)
# Train the model on the new data for a few epochs
history_F = model.fit(x = X, y = y,
                    batch_size = 4,
                    epochs = n_epochs,
                    validation_split = 0, 
                    shuffle = True,
                    # callbacks = [early_stopping_]
                    validation_data=(X_test,y_test)
                    #validation_data=(X,y)
                     )  

model.evaluate(X_test,y_test)

# acc_list=[]
# for jj in range(len(X_test)):
#     y_pred = model.predict(X_test[jj][None,...])
#     y_pred = (y_pred[0,:,:,0] > 0.9)
#     temp=np.equal(y_pred,y_test[jj])
#     score=np.sum(temp)
#     accuracy=score/(224**2)
#     acc_list.append(accuracy)

# acc_nump=np.array(acc_list)
# np.mean(acc_nump)

# iou_list=[]
# for jj in range(len(X_test)):
#     y_pred = model.predict(X_test[jj][None,...])
#     iou=class_IoU(y_test[jj], y_pred[0,:,:,0])
#     iou_list.append(iou.numpy())

# iou_list=np.array(iou_list)
# print("iou greater 0.4 and %")
# a1=len(np.where(iou_list>0.4)[0])
# print(a1,a1/500)
# print("iou greater 0.5 and %")
# a1=len(np.where(iou_list>0.5)[0])
# print(a1,a1/500)
# print("iou greater 0.6 and %")
# a1=len(np.where(iou_list>0.6)[0])
# print(a1,a1/500)
# print("iou greater 0.7 and %")
# a1=len(np.where(iou_list>0.7)[0])
# print(a1,a1/500)

#Saving the model
cwd = os.getcwd()
model_path = os.path.join(cwd,'bin','mobileNetv3_segmentation_new1.h5')
model.save(model_path, save_format='h5')