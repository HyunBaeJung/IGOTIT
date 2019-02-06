# coding: utf-8

# In[6]:


import speech_recognition as sr

def stt():
    
    # Convert `data` to 32 bit integers:

    r = sr.Recognizer()


    with sr.Microphone() as source:
        print('Say something!')
        audio = r.listen(source)


    try:
        text = r.recognize_google(audio)
        print('you said:{}'.format(text))
        
        #return text
        
    except Exception as e:
        print(e)


# In[7]:


def texting():
    
    isEnd = True
    
    while isEnd:
        stt()


if __name__ == "__main__":
    recording()