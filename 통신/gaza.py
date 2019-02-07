
# coding: utf-8

# In[2]:


import SR_module, PD_module, buffering, acceleration


# In[3]:


from threading import Thread


# In[4]:


def test():
    while True:
        buffer=buffering.buffering()
        if SR_module.main(buffer)=='scream':
            if PD_module.percent_scream(PD_module.model_predict('igotit.h5',PD_module.data_process()))==True:
                acceleration.scream_situation()
        elif SR_module.main(buffer)=='clash':
            if PD_module.percent_clash(PD_module.model_predict('igotit.h5',PD_module.data_process()))==True:
                PD_module.acceleration()
                acceleration.clash_situation()
        else:
            SR_module.message()        
#     for delfile in buffer:
#         os.remove("test/"+"Buf_"+delfile)


# In[5]:



def main():
    
    global cont
    
    p1 = Thread(target=acceleration.serveropne) #함수 1을 위한 프로세스
    p2 = Thread(target=test) #함수 1을 위한 프로세스
    
    # start로 각 프로세스를 시작합니다. func1이 끝나지 않아도 func2가 실행됩니다.
    p1.start()
    p2.start()
                        

    # join으로 각 프로세스가 종료되길 기다립니다 p1.join()이 끝난 후 p2.join()을 수행합니다
    p1.join()
    p2.join()


# In[ ]:


main()


# In[ ]:


# buffer=buffering.buffering()
# if SR_module.main(buffer)=='scream':
#     if PD_module.percent_scream(PD_module.model_predict('igotit.h5',PD_module.data_process()))==True:
acceleration.scream_situation
# elif SR_module.main(buffer)=='clash':
#     if PD_module.percent_clash(PD_module.model_predict('igotit.h5',PD_module.data_process()))==True:
#         PD_module.acceleration()
# else:
#     SR_module.message()        
#     for delfile in buffer:
#         os.remove("test/"+"Buf_"+delfile)


# In[ ]:


def percent_scream(list):
    
    per={}
    for a in list:
        if a in per:
            per[a] = per[a]+1
        else:
            per[a] = 1
    scream=0
    for i in list:
        if i==1:
            scream+=1
    print("비명일 확률은 : %.2f%%입니다"%((scream/len(list))*100))
    if (scream/len(list)*100)>=80:
        return True

