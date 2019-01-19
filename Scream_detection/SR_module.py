#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


# In[2]:


#FFT(Fast Fourier Transform)는 아주 적은 계산량으로 
#이산 푸리에 변환값을 계산하는 알고리즘을 말한다. 
def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    # nomalize위한 코드
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)


# In[3]:


#소리 데이터의 무의미한 부분은 식병을 어렵게 하기 때문에
#소리의 역치를 이용해 그런 부분을 제거하는 함수
#signal envloperk rmfjsrjek.
#thresh hold값은 0.0005로 셋팅하는게 이분에겐 먹혔지만
#이상하게 나온다면 알아서들 조절해봐라.
def envelope(y, rate, threshold):
    mask=[]
    #가끔 마이너스감 절댓값
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# In[4]:


def main():
    f = open("test.csv", 'w')
    f.write("fname,label\n")
    f.write("test.wav,test")
    f.close()
    
    df = pd.read_csv('test.csv')
    df.set_index('fname', inplace=True)
    for f in df.index:
        rate, signal = wavfile.read('test/'+f)
        df.at[f, 'length'] = signal.shape[0]/rate
    df.reset_index(inplace=True)
    classes = list(np.unique(df.label))
    
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    
    for c in classes:
        wav_file = df[df.label == c].iloc[0,0]
        # librosa에 load하면 signal과 sr즉 sound rate가 발생됩니다.
        # signal은 음파를 행렬로 표현한 데이터이고
        # sr은 그 파일의 sound rate를 자동으로 뽑아준다.
        signal, rate = librosa.load('test/'+wav_file, sr =44100)
        mask = envelope(signal, rate, 0.0005)
        signal=signal[mask]
        signals[c] = signal
        fft[c]= calc_fft(signal,rate)

        bank = logfbank(signal[:rate],rate,nfilt=26,nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
        
    for f in tqdm(df.fname):
        signal, rate = librosa.load('test/'+f, sr=16000)
        mask = envelope(signal, rate, 0.05) #0,0005 였음
        wavfile.write(filename='test_clean/'+f, rate=rate, data= signal[mask])
    


# - die out되는 부분을 없애야 인식을 잘 할수 있다.
# - 그래서 필터링을 해야한다. noise 역치를 만든다.
# - dead space를 없애야한다. 매우 중요하다.

# # 지금까지의 Process
# - 오디오 파일을 불러온다
# - Noise Threshold를 이용해 무의미한 부분을 제거한다.
# - Down Scaling을 한다.
# - 정제된 데이터를 clean 폴더에 저장한다.

# In[ ]:




