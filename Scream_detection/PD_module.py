#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
from keras.models import load_model


# In[2]:


def build_rand_feat(n_samples,class_dist,prob_dist,df,config,classes):
    x = []
    y = []
    _min, _max = float('inf'),-float('inf')
    #뉴럴 네트워크에선 보통 결과값을 0~1로 정제하기떄문에
    #min max값을 아는것이 중요하다.
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('test_clean/'+file)
        label = df.at[file,'label']
        rand_index = np.random.randint(0,wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        x_sample = mfcc(sample, rate,
                       numcep=config.nfeat, nfft=config.nfft).T
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amin(x_sample), _max)
        x.append(x_sample if config.mode == 'conv' else x_sample.T)
        y.append(classes.index(label))
    x, y = np.array(x), np.array(y)
    x = (x- _min) / (_max - _min)
    if config.mode == 'conv':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    elif config.mode == 'time':
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    y = to_categorical(y, num_classes=10)
    
    return x,y


# 머신러닝시 알아야될 값들 class 형식으로

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat = 13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.rate=rate
        self.step = int(rate/10)


# 데이터 프로세스

def data_process():
    df = pd.read_csv('test.csv')
    df.set_index('fname', inplace=True)
    
    for f in df.index:
        rate, signal = wavfile.read('test_clean/'+f)
        df.at[f, 'length'] = signal.shape[0]/rate


    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    #10분의 1로 데이터를 자름
    n_samples = 2 * int(df['length'].sum()/0.1)
    prob_dist = class_dist / class_dist.sum()
    #카테고리 하나 랜덤으로 뽑아옴
    choices = np.random.choice(class_dist.index, p= prob_dist)
    
    config = Config(mode='conv')
    
    x,y = build_rand_feat(n_samples,class_dist,prob_dist,df,config,classes)
    y_flat = np.argmax(y, axis=1)
    input_shape = (x.shape[1], x.shape[2], 1)
    
    return x


# 모델 로드하고 예측값 뽑기

def model_predict(model_name, pd_data):
    model = load_model(model_name)
    pre=model.predict_classes(pd_data)
    
    print(pre)
    
    return pre

# 예측값 받은파일 %로 나타내주기

def percent(list):
    
    per={}
    
    for a in list:
        if a in per:
            per[a] = per[a]+1
        else:
            per[a] = 1  
    
    print("비명일 확률은 : "+str(np.round(per[0]/len(list)*100,2))+"% 입니다.")
    

