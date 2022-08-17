from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf
from sklearn import metrics
from sklearn.cluster import KMeans
import time
from tensorflow.keras.preprocessing.image import array_to_img
import sklearn
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
import random
from tensorflow import keras
import tensorflow
import math
from tensorflow.keras.initializers import Constant
from tensorflow.python.keras.utils import tf_utils

# Compute EER metric based on TPR and FPR from predictions
def compute_eer_labels(labels, y_score):

    pred = [0 if x==1 else 1 for x in labels]

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(pred, y_score, pos_label=1)
    fnr = 1 - tpr

    t = np.nanargmin(np.abs(fnr-fpr))
    eer_low, eer_high = min(fnr[t],fpr[t]), max(fnr[t],fpr[t])
    eer = (eer_low+eer_high)*0.5

    return eer

#compute log llK score of being a bonafide audio 
def compute_scores(predictions, inverse=False):
    # predictions to scores (predictions are for spoof class with softmax/sigmoid when inverse is False)
    predictions = predictions.reshape(-1,1) # new to fix new version
    if inverse:
        loglikelihood = np.concatenate((np.log(1. - predictions+0.00000001), np.log(predictions+0.00000001)), axis=1)
    else:
        loglikelihood = np.concatenate((np.log(predictions+0.00000001), np.log(1. - predictions+0.00000001)), axis=1)
    scores = loglikelihood[:, 1] - loglikelihood[:, 0] #for bonafide
    return scores

#compute EER metric (equal error rate).
def compute_eer(labels, predictions, inverse=False):
    # predictions to scores (predictions are for spoof class when inverse is False)
    scores = compute_scores(predictions, inverse=inverse)
    EER = compute_eer_labels(labels, scores)
    return EER

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
                    gamma_1_sample = tensorflow.random.gamma(shape=[size], alpha=concentration_1)
                    gamma_2_sample = tensorflow.random.gamma(shape=[size], alpha=concentration_0)
                    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
            
#Main training method. Called after constructing the model architecture.
def train_CNN(model, x_train_normal, x_train_anomaly, model_path, class_=2, batch_size=32, num_epochs=400, initial_learning_rate = 0.0001, arcFace=False, label_smoothing=False, smooth_alpha=0.15, cosineRestart=True, mixup=False, shuffleAugment=False, new_mixup=False, shuffle_extended=False, freq_shuffling=False, attack_mixup=False, alpha_attackMixup=1.0, alpha_mixup=1.0):
    acc = []
    epochs = num_epochs
    if cosineRestart:
        decay_steps = num_epochs #400
        #Use cosine decay to 5% of initial learning_rate over the entire epochs.
        lr_decayed_fn = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate, first_decay_steps=40, alpha=5/100)
    print("training...")
    
    if mixup:
        batch_size = 2 * batch_size
    
    for epochnumber in range(epochs):
        if cosineRestart:
            # calculate learning rate:
            current_learning_rate = lr_decayed_fn(epochnumber).numpy()
            K.set_value(model.optimizer.learning_rate, current_learning_rate)  # set new learning_rate
        temp_acc = []
        
        # data shuffle
        np.random.shuffle(x_train_normal)
        np.random.shuffle(x_train_anomaly)
    
        for i in range(int(len(x_train_normal) / batch_size)):
            
            # load batch
            if epochnumber < 40:
                x_batch = x_train_normal[i*batch_size:i*batch_size+batch_size]
                x_batch_random = x_train_anomaly[i*batch_size:i*batch_size+batch_size]
                if attack_mixup:
                    indices = range(x_train_anomaly.shape[0])
                    indices = random.sample(indices, batch_size)
                    x_batch_random_mixup = x_train_anomaly[indices]
                    x_batch_random_mixup = np.vstack((x_batch_random, np.array(x_batch_random_mixup)))
                    X_shape_mixup = x_batch_random_mixup.shape
                    alphas = sample_beta_distribution(size=(X_shape_mixup[0]//4), concentration_0=alpha_attackMixup, concentration_1=alpha_attackMixup).numpy()
                    alphas = np.concatenate((np.zeros(X_shape_mixup[0]//4), alphas)).reshape(-1,1)                    
                    x_batch_random_mixup = x_batch_random_mixup.reshape(X_shape_mixup[0],-1)
                    x_batch_random_mixup = alphas * x_batch_random_mixup[:(X_shape_mixup[0]//2),:] + (1-alphas) * x_batch_random_mixup[(X_shape_mixup[0]//2):,:]
                    x_batch_random = x_batch_random_mixup.reshape(((X_shape_mixup[0]//2), X_shape_mixup[1], X_shape_mixup[2], X_shape_mixup[3]))
                 
                    
                X = np.vstack((x_batch, np.array(x_batch_random)))
            else:
                x_batch = x_train_normal[i*batch_size:i*batch_size+batch_size]
                # Hard sampling: 
                # select the hardest spoof samples (high uncertainty of the model=low output probability)
                x_batch_random = x_train_anomaly[i*batch_size:i*batch_size+batch_size]

                if attack_mixup:
                    indices = range(x_train_anomaly.shape[0])
                    indices = random.sample(indices, batch_size)
                    x_batch_random_mixup = x_train_anomaly[indices]
                    x_batch_random_mixup = np.vstack((x_batch_random, np.array(x_batch_random_mixup)))
                    X_shape_mixup = x_batch_random_mixup.shape
                    alphas = sample_beta_distribution(size=(X_shape_mixup[0]//4), concentration_0=alpha_attackMixup, concentration_1=alpha_attackMixup).numpy()
                    alphas = np.concatenate((np.zeros(X_shape_mixup[0]//4), alphas)).reshape(-1,1)
                    x_batch_random_mixup = x_batch_random_mixup.reshape(X_shape_mixup[0],-1)
                    x_batch_random_mixup = alphas * x_batch_random_mixup[:(X_shape_mixup[0]//2),:] + (1-alphas) * x_batch_random_mixup[(X_shape_mixup[0]//2):,:]
                    x_batch_random = x_batch_random_mixup.reshape(((X_shape_mixup[0]//2), X_shape_mixup[1], X_shape_mixup[2], X_shape_mixup[3]))
                X = np.vstack((x_batch, np.array(x_batch_random)))

            X_shape = X.shape
            # make labels
            y_batch = np.zeros(X_shape[0])
            y_batch[(X_shape[0]//2):] = 1

            Y = tf.keras.utils.to_categorical(y_batch) #To put back
            if label_smoothing:
                Y = (1-smooth_alpha) * Y + smooth_alpha/class_
                
            ############ Mix up
            if shuffleAugment: # Put after mixup for efficiency since will deal with half data. But not sure if this is best config ??
                X = time_shuffle_array(X, shuffle_extended, freq_shuffling)
                
            if mixup:
                alphas = sample_beta_distribution(size=(X_shape[0]//4), concentration_0=alpha_mixup, concentration_1=alpha_mixup).numpy()
                alphas = np.concatenate((np.ones(X_shape[0]//8), alphas[(X_shape[0]//8):], np.zeros(X_shape[0]//8), alphas[:(X_shape[0]//8)])).reshape(-1,1)
                
                Y = alphas * Y[:(X_shape[0]//2),:] + (1-alphas) * Y[(X_shape[0]//2):,:]               
                if not new_mixup: #normal mixup
                    X = X.reshape(X_shape[0],-1)
                    X = alphas * X[:(X_shape[0]//2),:] + (1-alphas) * X[(X_shape[0]//2):,:]
                    X = X.reshape(((X_shape[0]//2), X_shape[1], X_shape[2], X_shape[3]))
                else:
                    X = prepare_mixup_array(X[:(X_shape[0]//2),:,:,:], X[(X_shape[0]//2):,:,:,:], alphas)

            if arcFace:
                loss = model.train_on_batch([X, Y], Y) # For ArcFace loss layer
                score = model.evaluate([X, Y], Y) # For ArcFace loss layer
            else:
                loss = model.train_on_batch(X, Y)
                score = model.evaluate(X, Y)
            temp_acc.append(score[1])

        acc.append(np.mean(temp_acc))
            
        if (epochnumber+1) % 20 == 0: #Save model
            print("epoch:",epochnumber+1)
            print("Training accuracy:", acc[-1])
            model_path_epoch = os.path.join(model_path, str(epochnumber+1)+'_epochs')
            if not os.path.exists(model_path_epoch):
                os.makedirs(model_path_epoch)
            model.save(model_path_epoch)

    # Finally: plot accuracy graph over epochs
    plt.plot(acc,label="Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    return model

def time_mixup(x_train1, x_train2, alpha):
    indices = np.arange(x_train1.shape[1])
    size_zero = int((1 - alpha) * x_train1.shape[1])
    #print(size_zero, alpha)
    zero_indices = np.random.choice(indices, size=size_zero)
    if np.random.randint(1, 3) == 1: 
        x_train1[:,zero_indices] = x_train2[:,zero_indices] #Random bins order for 2nd audio
    else:
        x_train1[:,zero_indices] = x_train2[:,:size_zero] #keep order of original time bins
    return x_train1

def prepare_mixup_array(x_train1, x_train2, alphas):
    result = []
    X_shape = x_train1.shape
    #print(X_shape)
    for i, x1 in enumerate(x_train1):
        x1 = x1.reshape((X_shape[1], X_shape[2]))
        x2 = x_train2[i].reshape((X_shape[1], X_shape[2]))
        result.append(time_mixup(x1, x2, alphas[i]))
    result = np.array(result).reshape((X_shape[0], X_shape[1], X_shape[2], X_shape[3]))
    return result

def create_model_AM_logits(backbone_model, optimizer='Adam'):
    v2 = Model(inputs=backbone_model.input, outputs=backbone_model.layers[-2].output)
    v2 = build_arcface((2,224,224,1), v2, layer="AM_logits", optimizer=optimizer)
    return v2

# concatenate mobilenetV2 with arcface loss layer
def build_arcface(x, base_model, layer="", optimizer='Adam'):
    #add new layers 
    hidden = base_model.output
    yinput = Input(shape=(2,), name="am_logits_input") # for ArcFace
    # stock hidden model
    if layer == "AM_logits":
        c = AM_logits(2, 30, 0.35)([hidden,yinput])
    
    prediction = Activation('softmax')(c)
    model = Model(inputs=[base_model.input, yinput], outputs=prediction)

    if optimizer=='SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

#https://github.com/4uiiurz1/keras-arcface/blob/master/metrics.py
def resolve_training_flag(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    if tf.__version__ == "2.8.0" or tf.__version__ == "2.8.0-dev20211222" or tf.__version__ == "2.4.0":
        return tf.constant(training)
    return tf_utils.constant_value(training)

class AM_logits(Layer):
    """
    Implementation of CosFace layer. Reference: https://arxiv.org/abs/1801.09414 NOO
    
    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
      
      
    loss head proposed in paper:<Additive Margin Softmax for Face Verification>
    link: https://arxiv.org/abs/1801.05599
    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes 
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.35,
                 regularizer=None,
                 name='AM_logits',
                 **kwargs):

        super(AM_logits, self).__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer
    def get_config(self): #added abderrahim
        config = super().get_config()
        config.update({
            "_n_classes": self._n_classes,
            "_s": self._s,
            "_m": self._m,
            "_regularizer": self._regularizer,
        })
        return config

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')
    @tf.function # for >= tensorflow 2.2
    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights') #try epsilon=1e-10 ?
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')
        cosine_sim = tf.clip_by_value(cosine_sim, -1,1) # for numerical steady

        training = resolve_training_flag(self, training)

        if training:
            one_hot_labels = label
            final_theta = tf.where(tf.cast(one_hot_labels, dtype=tf.bool),
                                   cosine_sim - self._m,
                                   cosine_sim,
                                   name='cosine_sim_with_margin')
            return self._s * final_theta
        else:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim