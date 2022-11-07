import os
import cv2
import glob
import tqdm
import numpy as np

def loadData(imgList, maskList):
    X = []
    y = []
    for i in imgList:
            img_name = os.path.split(i)[1].split('.jpg')[0]#.split('_')[1]           
            img = cv2.imread(i)
            X.append(img)
    for i in maskList:
            mask_name = os.path.split(i)[1].split('.jpg')[0]#.split('_')[1]
            mask = cv2.imread(i,0)
            y.append(mask)
    return (np.array(X), np.array(y))

def custom_shuffle(img_array, mask_array):
    assert len(img_array) == len(mask_array)
    p = np.random.permutation(len(img_array))
    return img_array[p], mask_array[p]

def resizeImage(X, width, force_dim=False, height = None):
    X_res = []
    for img in X:
        r = width / img.shape[1]
        dim = (width, int(img.shape[0] * r))
        # not mantain the ratio
        if force_dim:
            img_resized = cv2.resize(img.astype('uint8'), (width,height), interpolation = cv2.INTER_AREA)
        else:
            img_resized = cv2.resize(img.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
        X_res.append(img_resized)
        
    return np.array(X_res)

#Resize the prediction mantaining the aspect ratio for y
def resizeImage_y(y, width, force_dim=False, height = None):
    y_res = []
    for img,img1 in tqdm(zip(y[:,0],y[:,1])):
        
        
        r = width / img.shape[1]
        dim = (width, int(img.shape[0] * r))
        # not mantain the ratio
        if force_dim:
            img_resized = cv2.resize(img.astype('uint8'), (width,height), interpolation = cv2.INTER_AREA)
        else:
            img_resized = cv2.resize(img.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
            
        r = width / img1.shape[1]
        dim = (width, int(img1.shape[0] * r))
        # not mantain the ratio
        if force_dim:
            img_resized1 = cv2.resize(img1.astype('uint8'), (width,height), interpolation = cv2.INTER_AREA)
        else:
            img_resized1 = cv2.resize(img1.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
            
            
        y_res.append([img_resized,img_resized1])
    
    return np.array(y_res)

def normalize(X):
    return (X / 255)

def get_data(PATH_DIR = 'dataset/aghi_mod/',in_net_h=480,in_net_w=640):
    training_img_dir = os.path.join(PATH_DIR, 'images/train')
    training_mask_dir = os.path.join(PATH_DIR, 'masks/train')
    test_img_dir = os.path.join(PATH_DIR, 'images/test')
    test_mask_dir = os.path.join(PATH_DIR, 'masks/test')
    train_img_list = glob.glob(os.path.join(training_img_dir, '*.jpg'))
    train_mask_list = glob.glob(os.path.join(training_mask_dir, '*.jpg'))
    test_img_list = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    test_mask_list = glob.glob(os.path.join(test_mask_dir, '*.jpg'))

    #X,y are for training and validation
    X, y = loadData(train_img_list, train_mask_list)
    #X_test,y_test are for testing
    X_test, y_test = loadData(test_img_list, test_mask_list)
    #mask and image must be shuffled with the same index
    X,y=custom_shuffle(X,y)
    X_test,y_test=custom_shuffle(X_test,y_test)
    # X = resizeImage(X, in_net_w, force_dim=True, height = in_net_h)
    # X_test = resizeImage(X=X_test,width= in_net_w, force_dim=True, height = in_net_h)
    # y = resizeImage(y, in_net_w, force_dim=True, height = in_net_h)
    # y_test = resizeImage(X=y_test, width=in_net_w, force_dim=True, height = in_net_h)
    X = normalize(X)
    X_test = normalize(X=X_test)    
    y= normalize(y)
    y_test= normalize(X=y_test)
    #Apply a threshold to have 0,1 despite the resizing
    y[y>=(0.5)]=1.0    
    y[y<(0.5)]=0.0
    #Apply a threshold to have 0,1 despite the resizing
    y_test[y_test>=(0.5)]=1.0
    y_test[y_test<(0.5)]=0.0
    # print(X.shape)
    return X,y,X_test,y_test