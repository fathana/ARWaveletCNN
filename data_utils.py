import tarfile, shutil
import zipfile
import sys, os, urllib.request, tarfile, glob
import numpy as np
import cv2
import librosa
import librosa.core
import librosa.feature
import librosa.display 
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tensorflow import keras
import logging
import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import audiofile as af
import librosa
import scipy
import scipy.fftpack
import librosa
from spafe.spafe.features.log_linear_fbanks import log_linear_fbanks

def store_file(test_score_file, test_filenames, test_scores):
    test_file = open(test_score_file, "w")
    for index in range(len(test_filenames)):
        test_file.write(str(test_filenames[index]) + " " + str(test_scores[index]) + "\n")
    test_file.close()
    print("DONE.")
    
def normalization(x, max_=None, min_=None, mean=None, sigma=None, resize=True, isNormalize=True):
    if resize:
        result = cv2.resize(x, (224, 224))
    else:
        result = x
    if not isNormalize:
        return result
    result = (result-min_)/(max_-min_)
    return (result*255-mean)/sigma

#create total dataset with labels. This is used for the ASV19 downstream task evaluation.
def create_data_downstream_labels(x_normal, x_anomaly, max_=None, min_=None, mean=None, sigma=None, features='stft', isNormalize=True):
    X_total = np.concatenate((x_normal, x_anomaly))
    # make labels
    y_total = np.zeros(len(X_total), dtype=int)
    y_total[len(x_normal):] = 1
        
    if features=='fbanks_delta_doubledelta' or features=='HPSS_power' or features=='HPSS_stft' or features=='HPSS_llfb' or features=='HPSS_decompose_audio' or features=='HPSS_llfb_doubledelta':
        X_total = create_data_downstream_delta(X_total, max_, min_, mean, sigma, features, isNormalize)
    else:
        #convert and normalize data
        X_total = create_data_downstream(X_total, max_, min_, mean, sigma, features, isNormalize)
    return X_total, y_total

# Iterate through dataset by converting audio waveforms into 224x224 stft spectrogram-images, then normalize the data according to the training set.
# stft stands for Short-time Fourier transform (https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
# (max_, min_, mean, sigma) from the bonafide training dataset statistics
def create_data_downstream(x_data, max_=None, min_=None, mean=None, sigma=None, features='stft', isNormalize=True): 
    X_batch = []
    for j in range(len(x_data)):
        #convert to a 224x224 stft spectrogram-image
        img = to_img2(x_data[j], features)
        #normalize data
        if features not in ['fbanks_delta_doubledelta', 'HPSS_power', 'HPSS_stft', 'HPSS_llfb', 'HPSS_decompose_audio', 'HPSS_llfb_doubledelta']:
            img = normalization(img, max_, min_, mean, sigma, resize=True, isNormalize=isNormalize)
        else:
            img = normalization(img, max_, min_, mean, sigma, resize=False, isNormalize=isNormalize)
        X_batch.append(img) 
    x_data = np.array(X_batch)
    if features not in ['fbanks_delta_doubledelta', 'HPSS_power', 'HPSS_stft', 'HPSS_llfb', 'HPSS_decompose_audio', 'HPSS_llfb_doubledelta']:
        x_data = np.expand_dims(x_data, axis=-1)
    return x_data

def create_data_downstream_delta(x_data, max_=None, min_=None, mean=None, sigma=None, features='stft', isNormalize=True): 
    X_batch = []
    for j in range(len(x_data)):
        #convert to a 224x224 stft spectrogram-image
        img = to_img2(x_data[j], features)
        #normalize data
        img = normalization(img, max_, min_, mean, sigma, resize=False, isNormalize=isNormalize)
        X_batch.append(img) 
    x_data = np.array(X_batch)
    return x_data

# load wave audio file.
def file_load(wav_name, mono=False):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))
        logger.error("file_broken or not exists!! : {}".format(wav_name))

#Load all audio files in a folder. waveform as output
def make_data(folder_name):
    result = []
    all_name = glob.glob(folder_name)
    for name in all_name:
        result.append(file_load(name)[0])
    return np.array(result)

def to_img(x, features='stft'):
    result = []
    if features=='HPSS_power':
        lin_filter=lin(16000, 512)
    for i in range(len(x)):
        if features=='fbanks':
            result.append(cv2.resize(to_log_linear_fbanks(x[i]), (224,224)))
        elif features=='fbanks_delta_doubledelta':
            llfb = to_log_linear_fbanks(x[i])
            llfb_delta = librosa.feature.delta(llfb)
            llfb_delta2 = librosa.feature.delta(llfb, order=2)
            # 3 channels
            feature = concatenate_feat(llfb, llfb_delta, llfb_delta2)
            result.append(feature)
        elif features=='fbanks_delta':
            result.append(cv2.resize(librosa.feature.delta(to_log_linear_fbanks(x[i])), (224,224)))   
        elif features=='fbanks_doubledelta':
            result.append(cv2.resize(librosa.feature.delta(to_log_linear_fbanks(x[i]), order=2), (224,224)))
        elif features=='HPSS_power':
            H_power, P_power, R_power = HPSS_power(x[i], lin_filter)
            # 3 channels
            feature = concatenate_feat(H_power, P_power, R_power)
            result.append(feature)
        elif features=='HPSS_stft':
            spec, H, P = HPSS(x[i], first_feature = "stft")
            # 3 channels
            feature = concatenate_feat(spec, H, P)
            result.append(feature)
        elif features=='HPSS_llfb':
            llfb, H, P = HPSS(x[i], first_feature = "fbanks")
            # 3 channels
            feature = concatenate_feat(llfb, H, P)
            result.append(feature)
        elif features=='HPSS_llfb_doubledelta':
            llfb, H, P = HPSS(x[i], first_feature = "fbanks")
            llfb = to_log_linear_fbanks(x[i])
            llfb_delta2 = librosa.feature.delta(llfb, order=2)
            # 3 channels
            feature = concatenate_feat(llfb, H, llfb_delta2)
            result.append(feature)
        elif features=='HPSS_decompose_audio':
            llfb, H_llfb, P_llfb = HPSS_decompose_audio(x[i])
            # 3 channels
            feature = concatenate_feat(llfb, H_llfb, P_llfb)
            result.append(feature)
        else:
            result.append(cv2.resize(to_sp(x[i]), (224,224)))
    return np.array(result)

def to_img2(x, features='stft'):
    if features=='HPSS_power':
        lin_filter=lin(16000, 512)
        
    if features=='fbanks':
        result = cv2.resize(to_log_linear_fbanks(x), (224,224))
    elif features=='fbanks_delta_doubledelta':
        llfb = to_log_linear_fbanks(x)
        llfb_delta = librosa.feature.delta(llfb)
        llfb_delta2 = librosa.feature.delta(llfb, order=2)
        # 3 channels
        feature = concatenate_feat(llfb, llfb_delta, llfb_delta2)
        result = feature
    elif features=='fbanks_delta':
        result = cv2.resize(librosa.feature.delta(to_log_linear_fbanks(x)), (224,224))   
    elif features=='fbanks_doubledelta':
        result = cv2.resize(librosa.feature.delta(to_log_linear_fbanks(x), order=2), (224,224))
    elif features=='HPSS_power':
        H_power, P_power, R_power = HPSS_power(x, lin_filter)
        # 3 channels
        feature = concatenate_feat(H_power, P_power, R_power)
        result = feature
    elif features=='HPSS_stft':
        spec, H, P = HPSS(x, first_feature = "stft")
        # 3 channels
        feature = concatenate_feat(spec, H, P)
        result = feature
    elif features=='HPSS_llfb':
        llfb, H, P = HPSS(x, first_feature = "fbanks")
        # 3 channels
        feature = concatenate_feat(llfb, H, P)
        result = feature
    elif features=='HPSS_llfb_doubledelta':
        llfb, H, P = HPSS(x, first_feature = "fbanks")
        llfb = to_log_linear_fbanks(x)
        llfb_delta2 = librosa.feature.delta(llfb, order=2)
        # 3 channels
        feature = concatenate_feat(llfb, H, llfb_delta2)
        result = feature
    elif features=='HPSS_decompose_audio':
        llfb, H_llfb, P_llfb = HPSS_decompose_audio(x)
        # 3 channels
        feature = concatenate_feat(llfb, H_llfb, P_llfb)
        result = feature
    else:
        result = cv2.resize(to_sp(x), (224,224))
    return np.array(result)
    
def lin(sr, n_fft, n_filter=128, fmin=0.0, fmax=None, dtype=numpy.float32):

    if fmax is None:
        fmax = float(sr) / 2

    n_filter = int(n_filter)
    weights = numpy.zeros((n_filter, int(1 + n_fft // 2)), dtype=dtype)

    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    linear_f = numpy.linspace(fmin, fmax, n_filter + 2)

    fdiff = numpy.diff(linear_f)
    ramps = numpy.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filter):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = numpy.maximum(0, numpy.minimum(lower, upper))

    return weights

def HPSS_power(clip, lin_filter):
    audio=librosa.effects.preemphasis(clip)
    spec=librosa.stft(audio, n_fft=512, hop_length=160, win_length=400)
    H, P=librosa.decompose.hpss(spec, margin=3.0)
    R=spec-(H+P)
    H_power=librosa.power_to_db(numpy.dot(lin_filter, H))
    P_power=librosa.power_to_db(numpy.dot(lin_filter, P))
    R_power=librosa.power_to_db(numpy.dot(lin_filter, R))

    return H_power, P_power, R_power
def HPSS(clip, first_feature = "stft"):
    audio=librosa.effects.preemphasis(clip)
    spec=librosa.stft(audio, n_fft=512, hop_length=160, win_length=400)
    H, P = librosa.decompose.hpss(spec)
    spec = librosa.amplitude_to_db(np.abs(spec))
    H = librosa.amplitude_to_db(np.abs(H))
    P = librosa.amplitude_to_db(np.abs(P))

    if first_feature == "stft":
        return spec, H, P
    else:
        return to_log_linear_fbanks(clip), H, P
def HPSS_decompose_audio(clip):
    # Get a more isolated percussive component by widening its margin
    clip_harmonic, clip_percussive = librosa.effects.hpss(clip)
    llfb = to_log_linear_fbanks(clip)
    H_llfb = to_log_linear_fbanks(clip_harmonic)
    P_llfb = to_log_linear_fbanks(clip_percussive)

    return llfb, H_llfb, P_llfb

# change wave data to stft (Short-time Fourier transform)
# possibility to change the window length during stft computation. Tested 25 ms and 20 ms. 32ms window size is the best.
# NFFT = 2^(ceil(log(winpts)/log(2)))
def to_sp(x, n_fft=512, hop_length=256): #, win_length=20):
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)#, win_length=win_length)
    #to log scale (for compression)
    sp = librosa.amplitude_to_db(np.abs(stft))
    return sp

def to_log_linear_fbanks(clip):
    log_lfbanks  = log_linear_fbanks(clip, fs=16000, num_ceps=40, nfilts=40, normalize=0)
    return log_lfbanks

def concatenate_feat(first, second, third):
    first = cv2.resize(first, (224,224))
    second = cv2.resize(second, (224,224))
    third = cv2.resize(third, (224,224))
    # 3 channels
    feature = np.concatenate((first.reshape(224,224,1), second.reshape(224,224,1), third.reshape(224,224,1)), axis=2)
    return feature