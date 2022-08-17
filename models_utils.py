import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
import os
import random
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
import tempfile
import matplotlib as mpl

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.applications.mobilenet import MobileNet

import pathlib
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image

import pandas as pd
import seaborn as sns

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l1

def create_model_all_cnn_l1Reg_noise_dropout(IMG_WIDTH, IMG_HEIGHT, optimizer='Adam', embedding_dim=256, num_channels=1, wavelet_transform=True, is_gaussian_noise=False, is_l1=False, dropout_value=0.3, model_type="VGG16_2", include_preprocessing=True):
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    if model_type == "VGG16_2":
        base_model = VGG16(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "EfficientNetV2S":
        from efficientnet_v2 import EfficientNetV2S
        base_model = EfficientNetV2S(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels), include_preprocessing=include_preprocessing)
    elif model_type == "EfficientNetV2M":
        from efficientnet_v2 import EfficientNetV2M
        base_model = EfficientNetV2M(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels), include_preprocessing=include_preprocessing)
    elif model_type == "EfficientNetV2L":
        from tensorflow.keras.applications import EfficientNetV2L
        from efficientnet_v2 import EfficientNetV2L
        base_model = EfficientNetV2L(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels), include_preprocessing=include_preprocessing)
    elif model_type == "ResNet152V2":
        from tensorflow.keras.applications import ResNet152V2
        base_model = ResNet152V2(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "ResNet50":
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "seresnet18" or model_type == "seresnext50":
        from classification_models.classification_models.tfkeras import Classifiers
        base_model, preprocess_input = Classifiers.get(model_type)
        base_model = base_model(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "InceptionResNetV2":
        from tensorflow.keras.applications import InceptionResNetV2
        base_model = InceptionResNetV2(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "NASNetLarge":
        from tensorflow.keras.applications import NASNetLarge
        base_model = NASNetLarge(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "NASNetMobile":
        from tensorflow.keras.applications import NASNetMobile
        base_model = NASNetMobile(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    elif model_type == "Xception":
        from tensorflow.keras.applications import Xception
        base_model = Xception(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))

    if is_l1:
        kernel_reg = l1(0.0001)
    else:
        kernel_reg = None
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    fc_5 = Dense(2048, name='fc_5', kernel_regularizer=kernel_reg)(x) #(flat_5_1)
    norm_5 = BatchNormalization(name='norm_5')(fc_5)
    relu_5 = Activation('relu', name='relu_5')(norm_5)
    drop_5 = Dropout(dropout_value, name='drop_5')(relu_5)

    fc_6 = Dense(2048, name='fc_6', kernel_regularizer=kernel_reg)(drop_5)
    norm_6 = BatchNormalization(name='norm_6')(fc_6)
    relu_6 = Activation('relu', name='relu_6')(norm_6)
    drop_6 = Dropout(dropout_value, name='drop_6')(relu_6)

    output = Dense(2, activation='softmax', name='dense_7')(drop_6)

    model = Model(inputs=base_model.input, outputs=output)

    if optimizer=='SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def create_model_wavelet_cnn_l1Reg_noise_dropout(IMG_WIDTH, IMG_HEIGHT, optimizer='Adam', embedding_dim=256, num_channels=1, wavelet_transform=True, is_gaussian_noise=False, is_l1=False, dropout_value=0.3):
    input_shape = IMG_WIDTH, IMG_HEIGHT, num_channels

    input_ = Input(input_shape, name='the_input')
    print("#fixed wavelet")
    if num_channels==1:
        wavelet = Lambda(Wavelet, Wavelet_out_shape, name='wavelet')
    elif num_channels==3:
        wavelet = Lambda(Wavelet_3channels, Wavelet_out_shape, name='wavelet')
        
    if wavelet_transform:
        input_l1, input_l2, input_l3, input_l4 = wavelet(input_)
    else:
        input_l1, input_l2, input_l3, input_l4 = input_, input_, input_, input_
        
    if is_l1:
        kernel_reg = l1(0.0001)
    else:
        kernel_reg = None

    # level one decomposition starts
    conv_1 = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_1', kernel_regularizer=kernel_reg)(input_l1)
    norm_1 = BatchNormalization(name='norm_1')(conv_1)
    relu_1 = Activation('relu', name='relu_1')(norm_1)

    if is_gaussian_noise:
        relu_1 = GaussianNoise(stddev=0.1)(relu_1)
    conv_1_2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_1_2', kernel_regularizer=kernel_reg)(relu_1)
    norm_1_2 = BatchNormalization(name='norm_1_2')(conv_1_2)
    relu_1_2 = Activation('relu', name='relu_1_2')(norm_1_2)

    # level two decomposition starts
    conv_a = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_a', kernel_regularizer=kernel_reg)(input_l2)
    norm_a = BatchNormalization(name='norm_a')(conv_a)
    relu_a = Activation('relu', name='relu_a')(norm_a)

    # concate level one and level two decomposition
    concate_level_2 = concatenate([relu_1_2, relu_a])
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_2', kernel_regularizer=kernel_reg)(concate_level_2)
    norm_2 = BatchNormalization(name='norm_2')(conv_2)
    relu_2 = Activation('relu', name='relu_2')(norm_2)
    
    if is_gaussian_noise:
        relu_2 = GaussianNoise(stddev=0.1)(relu_2)
    conv_2_2 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_2_2', kernel_regularizer=kernel_reg)(relu_2)
    norm_2_2 = BatchNormalization(name='norm_2_2')(conv_2_2)
    relu_2_2 = Activation('relu', name='relu_2_2')(norm_2_2)

    # level three decomposition starts 
    conv_b = Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_b', kernel_regularizer=kernel_reg)(input_l3)
    norm_b = BatchNormalization(name='norm_b')(conv_b)
    relu_b = Activation('relu', name='relu_b')(norm_b)

    if is_gaussian_noise:
        relu_b = GaussianNoise(stddev=0.1)(relu_b)
    conv_b_2 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_b_2', kernel_regularizer=kernel_reg)(relu_b)
    norm_b_2 = BatchNormalization(name='norm_b_2')(conv_b_2)
    relu_b_2 = Activation('relu', name='relu_b_2')(norm_b_2)

    # concate level two and level three decomposition 
    concate_level_3 = concatenate([relu_2_2, relu_b_2])
    conv_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_3', kernel_regularizer=kernel_reg)(concate_level_3)
    norm_3 = BatchNormalization(name='nomr_3')(conv_3)
    relu_3 = Activation('relu', name='relu_3')(norm_3)

    if is_gaussian_noise:
        relu_3 = GaussianNoise(stddev=0.1)(relu_3)
    conv_3_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_3_2', kernel_regularizer=kernel_reg)(relu_3)
    norm_3_2 = BatchNormalization(name='norm_3_2')(conv_3_2)
    relu_3_2 = Activation('relu', name='relu_3_2')(norm_3_2)

    # level four decomposition start
    conv_c = Conv2D(64, kernel_size=(3, 3), padding='same', name='conv_c', kernel_regularizer=kernel_reg)(input_l4)
    norm_c = BatchNormalization(name='norm_c')(conv_c)
    relu_c = Activation('relu', name='relu_c')(norm_c)

    if is_gaussian_noise:
        relu_c = GaussianNoise(stddev=0.1)(relu_c)
    conv_c_2 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_2', kernel_regularizer=kernel_reg)(relu_c)
    norm_c_2 = BatchNormalization(name='norm_c_2')(conv_c_2)
    relu_c_2 = Activation('relu', name='relu_c_2')(norm_c_2)

    conv_c_3 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_c_3', kernel_regularizer=kernel_reg)(relu_c_2)
    norm_c_3 = BatchNormalization(name='norm_c_3')(conv_c_3)
    relu_c_3 = Activation('relu', name='relu_c_3')(norm_c_3)

    # concate level level three and level four decomposition
    concate_level_4 = concatenate([relu_3_2, relu_c_3])
    conv_4 = Conv2D(256, kernel_size=(3, 3), padding='same', name='conv_4', kernel_regularizer=kernel_reg)(concate_level_4)
    norm_4 = BatchNormalization(name='norm_4')(conv_4)
    relu_4 = Activation('relu', name='relu_4')(norm_4)

    if is_gaussian_noise:
        relu_4 = GaussianNoise(stddev=0.1)(relu_4)
    conv_4_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv_4_2', kernel_regularizer=kernel_reg)(relu_4)
    norm_4_2 = BatchNormalization(name='norm_4_2')(conv_4_2)
    relu_4_2 = Activation('relu', name='relu_4_2')(norm_4_2)

    conv_5_1 = Conv2D(128, kernel_size=(3, 3), padding='same', name='conv_5_1', kernel_regularizer=kernel_reg)(relu_4_2)
    norm_5_1 = BatchNormalization(name='norm_5_1')(conv_5_1)
    relu_5_1 = Activation('relu', name='relu_5_1')(norm_5_1)

    pool_5_1 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='same', name='avg_pool_5_1')(relu_5_1)
    flat_5_1 = Flatten(name='flat_5_1')(pool_5_1) 

    fc_5 = Dense(2048, name='fc_5', kernel_regularizer=kernel_reg)(flat_5_1)
    norm_5 = BatchNormalization(name='norm_5')(fc_5)
    relu_5 = Activation('relu', name='relu_5')(norm_5)
    drop_5 = Dropout(dropout_value, name='drop_5')(relu_5)

    fc_6 = Dense(2048, name='fc_6', kernel_regularizer=kernel_reg)(drop_5)
    norm_6 = BatchNormalization(name='norm_6')(fc_6)
    relu_6 = Activation('relu', name='relu_6')(norm_6)
    drop_6 = Dropout(dropout_value, name='drop_6')(relu_6)

    output = Dense(2, activation='softmax', name='dense_7')(drop_6)

    model = Model(inputs=input_, outputs=output)
    
    if optimizer=='SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,# amsgrad=True),
                  metrics=['accuracy'])
    return model

def create_model_VGG16_custom_3_l1Reg_noise_dropout(IMG_WIDTH, IMG_HEIGHT, optimizer='Adam', embedding_dim=256, num_channels=1, is_dropout=False, is_gaussian_noise=False, is_l1=False, dropout_value=0.2):
    inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    vgg_16 = VGG16(input_tensor=inputs ,weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, num_channels))
    
    if is_l1:
        kernel_reg = l1(0.0001)
    else:
        kernel_reg = None
    
    # add a global spatial average pooling layer
    x = vgg_16.output
    
    if is_gaussian_noise:
        x = GaussianNoise(stddev=0.1)(x)
    x = layers.Conv2D(64, 3, activation=None, kernel_regularizer=kernel_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if is_dropout:
        x = Dropout(dropout_value, name='drop_1')(x)
    
    x = layers.GlobalAveragePooling2D()(x)

    if is_gaussian_noise:
        x = GaussianNoise(stddev=0.1)(x)
    x = layers.Dense(embedding_dim, activation=None, kernel_regularizer=kernel_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if is_dropout:
        x = Dropout(dropout_value, name='drop_2')(x)
    
    predictions = layers.Dense(2, activation='softmax')(x)
    
    # this is the model we will train
    model = keras.Model(inputs=vgg_16.input, outputs=predictions)

    if optimizer=='SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,# amsgrad=True),
                  metrics=['accuracy'])
    return model

# batch operation usng tensor slice
def WaveletTransformAxisY(batch_img):
    odd_img  = batch_img[:,0::2]
    even_img = batch_img[:,1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H

def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:,:,::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:,::-1,...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:,::-1,...]
    return dst_L, dst_H

def Wavelet(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:,0]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3]
    transform_batch_l3 = K.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4]
    transform_batch_l4 = K.stack(wavelet_data_l4, axis=1)

    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = K.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = K.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])
    
    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]


def Wavelet_out_shape(input_shapes):
    return [tuple([None, 112, 112, 12]), tuple([None, 56, 56, 12]), 
            tuple([None, 28, 28, 12]), tuple([None, 14, 14, 12])]

def Wavelet_3channels(batch_image):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:,0]
    g = batch_image[:,1]
    b = batch_image[:,2]

    # level 1 decomposition
    wavelet_L, wavelet_H = WaveletTransformAxisY(r)
    r_wavelet_LL, r_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    r_wavelet_HL, r_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(g)
    g_wavelet_LL, g_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    g_wavelet_HL, g_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_L, wavelet_H = WaveletTransformAxisY(b)
    b_wavelet_LL, b_wavelet_LH = WaveletTransformAxisX(wavelet_L)
    b_wavelet_HL, b_wavelet_HH = WaveletTransformAxisX(wavelet_H)

    wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH, 
    g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
    b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
    transform_batch = K.stack(wavelet_data, axis=1)

    # level 2 decomposition
    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(r_wavelet_LL)
    r_wavelet_LL2, r_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    r_wavelet_HL2, r_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(g_wavelet_LL)
    g_wavelet_LL2, g_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    g_wavelet_HL2, g_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)

    wavelet_L2, wavelet_H2 = WaveletTransformAxisY(b_wavelet_LL)
    b_wavelet_LL2, b_wavelet_LH2 = WaveletTransformAxisX(wavelet_L2)
    b_wavelet_HL2, b_wavelet_HH2 = WaveletTransformAxisX(wavelet_H2)


    wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2, 
    g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
    b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
    transform_batch_l2 = K.stack(wavelet_data_l2, axis=1)

    # level 3 decomposition
    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(r_wavelet_LL2)
    r_wavelet_LL3, r_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    r_wavelet_HL3, r_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(g_wavelet_LL2)
    g_wavelet_LL3, g_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    g_wavelet_HL3, g_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL2)
    b_wavelet_LL3, b_wavelet_LH3 = WaveletTransformAxisX(wavelet_L3)
    b_wavelet_HL3, b_wavelet_HH3 = WaveletTransformAxisX(wavelet_H3)

    wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3, 
    g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
    b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
    transform_batch_l3 = K.stack(wavelet_data_l3, axis=1)

    # level 4 decomposition
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(r_wavelet_LL3)
    r_wavelet_LL4, r_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    r_wavelet_HL4, r_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(g_wavelet_LL3)
    g_wavelet_LL4, g_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    g_wavelet_HL4, g_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)

    #wavelet_L3, wavelet_H3 = WaveletTransformAxisY(b_wavelet_LL3) #original
    wavelet_L4, wavelet_H4 = WaveletTransformAxisY(b_wavelet_LL3)
    b_wavelet_LL4, b_wavelet_LH4 = WaveletTransformAxisX(wavelet_L4)
    b_wavelet_HL4, b_wavelet_HH4 = WaveletTransformAxisX(wavelet_H4)


    wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4, 
    g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
    b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
    transform_batch_l4 = K.stack(wavelet_data_l4, axis=1)

    decom_level_1 = K.permute_dimensions(transform_batch, [0, 2, 3, 1])
    decom_level_2 = K.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])
    decom_level_3 = K.permute_dimensions(transform_batch_l3, [0, 2, 3, 1])
    decom_level_4 = K.permute_dimensions(transform_batch_l4, [0, 2, 3, 1])
    
    return [decom_level_1, decom_level_2, decom_level_3, decom_level_4]