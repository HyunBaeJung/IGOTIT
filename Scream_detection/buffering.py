
# coding: utf-8

# In[3]:


#녹음관련 import
import pyaudio
import wave

#시간관련 import
import time
import os
import datetime


# In[4]:


def oneSecondRec(file_name):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "test/"+"Buf_"+file_name+".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

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


# In[5]:


def timenow():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d_%H%M%S')
    return nowDatetime


# In[6]:


def buffering():
    buffer = []
    
    isEnd = True
    #몇번 도는지 range안의 숫자를 바꾸면 됨
    
    while isEnd:

        file=oneSecondRec(timenow())

        buffer.append(file)

        if len(buffer) >5:
            delfile = buffer.pop(0)
            os.remove("test/"+"Buf_"+delfile)
            
    for delfile in buffer:
        os.remove("test/"+"Buf_"+delfile)
        
        


# In[8]:


#파라미터로 몇초까지 버퍼링을 할 것인지 결정
#buffering(5)

