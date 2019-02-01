

#SR_module.main()
#PD_module.percent(PD_module.model_predict('ver1.0.h5',PD_module.data_process()))


from threading import Thread

import SR_module, PD_module, buffering, recording, appconnector, stt


# In[2]:

import random, time

def test():
    while 1>0:
        a= random.randint(1,1000)
        b= random.randint(1,1000)
        
        print(a-b)
        time.sleep(1)

def main():
   
   global cont
   
   #p1 = Thread(target=buffering.buffering) #함수 1을 위한 프로세스
   p2 = Thread(target=stt.texting) #함수 1을 위한 프로세스
   #p3 = Thread(target=recording.recording) #함수 1을 위한 프로세스
   p4 = Thread(target=appconnector.openSoket) #함수 1을 위한 프로세스
   p5 = Thread(target=test)
   
   # start로 각 프로세스를 시작합니다. func1이 끝나지 않아도 func2가 실행됩니다.
   #p1.start()
   p2.start()
   #p3.start()
   p4.start()
   p5.start()
                       

   # join으로 각 프로세스가 종료되길 기다립니다 p1.join()이 끝난 후 p2.join()을 수행합니다
   #p1.join()
   p2.join()
   #p3.join()
   p4.join()
   p5.join()


# In[ ]:


main()

