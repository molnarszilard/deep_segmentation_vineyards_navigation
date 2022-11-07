import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Activation, Input, Add, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape,Dropout, Multiply, Flatten,UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy


from tensorflow.keras.metrics import MeanIoU

from tensorflow.keras.utils import get_custom_objects
#import network
from tensorflow.keras.applications import MobileNetV3Large

#Define custom activation function
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6

def buildModel(base_model, dropout_rate= 0.2, n_class=1,activation_number=41): 
    #1/8 resolution output    
    out_1_8= base_model.get_layer('re_lu_15').output #'activation_15'
    # print(out_1_8.shape)
    #1/16 resolution output    
    out_1_16= base_model.get_layer('re_lu_29').output #'activation_29'
    # print(out_1_16.shape)
    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)    
    layer_name_act="activation_head"+str(activation_number)
    x1 = Activation('relu',name=layer_name_act)(x1)
    activation_number+=1   
    # print(x1.shape) 
    # branch2
    s = x1.shape
    #custom average pooling2D
    x2 = AveragePooling2D(pool_size=(10, 10), strides=(5, 4),data_format='channels_last')(out_1_16)#MS - s=pool-size were (12,12), strides=(4, 5)
    x2 = Conv2D(128, (1, 1))(x2)
    layer_name_act="activation_head"+str(activation_number)
    x2 = Activation('sigmoid',name=layer_name_act)(x2)
    activation_number+=1
    s2 = x2.shape
    # print(x2.shape)
    x2 = UpSampling2D(size=(int(s[1]/s2[1]), int(s[2]/s2[2])),data_format='channels_last',interpolation="bilinear")(x2) #MS - size=(int(s[1]), int(s[2]))
    # print(x2.shape)
    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)
    # print(x3.shape)
    # multiply
    m1 = Multiply()([x1, x2])
    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1))(m1)
    # add
    m2 = Add()([m1, x3])
    #adding this UPsampling of factor 8
    m2 = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    # predictions 
    layer_name_act="activation_head"+str(activation_number)
    predictions = Activation('sigmoid',name=layer_name_act)(m2)
    activation_number+=1 
    # final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_model(in_net_w,in_net_h):
    #Defining dropout_r
    dropout_r=0.2
    activation_number=41
    #Base model mobile net
    model_base= MobileNetV3Large(input_shape=(in_net_h,in_net_w,3),
                        alpha=1.0,
                        minimalistic=False,
                        include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        classes=1,
                        pooling='avg',
                        dropout_rate=dropout_r)
    # model_base.summary()
    model=buildModel(model_base,dropout_r,1,activation_number)
    model.summary()
    return model