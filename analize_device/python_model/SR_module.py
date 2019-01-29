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

# 1.def calc_fft():
# calc_fft함수는 np.fft.rfftfreq를 이용해 고속 푸리에 변환을 하는 알고리즘이다.
# freq = np.fft.rfftfreq(n, d=1/rate)
# freq는 진동수로, np.fft.rfftfreq를 사용한다
# Y = abs(np.fft.rfft(y)/n) 
# Y는 1차원 n포인트 이산 푸리에 변환값의 절대값을 사용한다.

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

# pd.series인 y에 절대값을 씌우고 1초마다  rolling mean을 찾는다.
# 이때 rolling mean이라 함은 시계열 데이터를 정해진 특정시간동안의 평균을 계산함을의미한다.  
# min_periods=1가 의미하는 것은  minimum number of observations in window로 즉 window에 있는 최소한의 값을 의미한다. 
# 이는 각 창마다 적어도 1개이상의 observation이 있어야 한다는 설정을 나타낸다.
# threshold를 0.0005로 설정한 후,  만약 rolling의 평균이 0.0005보다 큰 경우에만 의미있는 소리로 판단하여 mask라는
# 빈 리스트에 append시키고, 그렇지 않을 경우 append시키지 않는다. 즉 소리가 거의 감지되지않는 부분은 다 잘라내는 문법이라고 볼 수 있다. 
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

# def first_classify():
#     a={}
#     for i in range(len(fft['test'][0])):
#         a[fft['test'][1][i]]=fft['test'][0][i]
#     for i in a:
#         if a[i]==fft['test'][0].max():
#             b=i
#     if b>1000:
#         deep_learning=True
#     else:
#         deep_learning=False
#     return [deep_learning,b]


# In[4]:


def main():
    f = open("test.csv", 'w')#test.csv파일을 연다
    f.write("fname,label\n")#해당 파일에 fname,label이라고 쓰고 한줄 내린다
    f.write("test.wav,test")#두번째 줄에 test.wav,test라고 쓴다.
    f.close()#파일을 닫는다
    
    df = pd.read_csv('test.csv')#test.csv파일을 읽어와서 df라는 변수에 넣는다.
    df.set_index('fname', inplace=True)#해당 파일의 index를 fname으로 설정한다.
    for f in df.index:# df.index에 있는 f는 test.wav밖에없다.
        rate, signal = wavfile.read('test/'+f)#test폴더 안에있는 test.wav파일을 wavfile.read로 읽어와서 rate, signal라는 변수에 담는다.
        df.at[f, 'length'] = signal.shape[0]/rate
    df.reset_index(inplace=True)#인덱스를 reset
    classes = list(np.unique(df.label))#classes는 ['test']인 리스트이다.
    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}
    
    for c in classes:
        wav_file = df[df.label == c].iloc[0,0]
        # librosa에 load하면 signal과 sr즉 sound rate가 발생됩니다.
        # signal은 음파를 행렬로 표현한 데이터이고
        # sr은 그 파일의 sound rate를 자동으로 뽑아준다.
        signal, rate = librosa.load('test/'+wav_file, sr =None) #sr은 sampling rate로 샘플링되는 속도이다.
        #librosa.load는 시계열데이터(시간데이터)를 실수 Series로 만들어준다. 원본파일 샘플링 속도를 따라가려면 sr=None, 이 사람은 44100을 사용
        #default값은 sr=22050이다
        mask = envelope(signal, rate, 0.0005)
        signal=signal[mask]
        signals[c] = signal
        fft[c]= calc_fft(signal,rate)

        bank = logfbank(signal[:rate],rate,nfilt=26,nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel
    a={}
    for i in range(len(fft['test'][0])):
        a[fft['test'][1][i]]=fft['test'][0][i]
    for i in a:
        if a[i]==fft['test'][0].max():
            b=i
    if b>1000:
        deep_learning=True
    else:
        deep_learning=False

    if deep_learning==True:
        for f in tqdm(df.fname):
            signal, rate = librosa.load('test/'+f, sr=16000)
            mask = envelope(signal, rate, 0.0005) #0,0005 였음
            wavfile.write(filename='test_clean/'+f, rate=rate, data= signal[mask])
    else:
        print('최다 빈도 주파수가 %f로 너무 낮습니다.'%b)
    return deep_learning

    


# - die out되는 부분을 없애야 인식을 잘 할수 있다.
# - 그래서 필터링을 해야한다. noise 역치를 만든다.
# - dead space를 없애야한다. 매우 중요하다.

# # 지금까지의 Process
# - 오디오 파일을 불러온다
# - Noise Threshold를 이용해 무의미한 부분을 제거한다.
# - Down Scaling을 한다.
# - 정제된 데이터를 clean 폴더에 저장한다.





