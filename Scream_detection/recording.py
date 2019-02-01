
# coding: utf-8

# In[1]:


#녹음관련 import
import pyaudio
import wave

#시간관련 import
import time
import os
import datetime


# In[2]:


def oneSecondRec(file_name):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 60
    WAVE_OUTPUT_FILENAME = "test/"+file_name+".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* normal recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return file_name+".wav" 


# In[3]:


def timenow():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d_%H%M%S')
    return nowDatetime


# In[4]:


def recording():
    
    isEnd = True
    
    while isEnd:
        oneSecondRec(timenow())

