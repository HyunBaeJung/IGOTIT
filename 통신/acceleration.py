import time
import threading
import socket
import re
import os
import pandas as pd
global temp
temp ='0'

def file_create_loop(list):
    if i<=4:
        file=pd.DataFrame(list)
        file.to_csv('acceleration/file'+str(i)+'.csv')
    else:
        os.remove('acceleration/file1.csv')
        os.rename('acceleration/file2.csv','file1.csv')
        os.rename('acceleration/file3.csv','file2.csv')
        os.rename('acceleration/file4.csv','file3.csv')
        file.to_csv('acceleration/file4.csv')

def decoding(a):
    a=a
    b=a.decode("utf-8")
    print(b)


def serveropne():
    global temp
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except socket.error as err :
        print("에러 발생 원인 :  %s"%(err))
    msg=bytearray(temp,'utf-8')
    HOST='192.168.22.106'
    port=1024
    s.bind((HOST,port))
    print("%d 포트로 연결을 기다리는중"%(port))
    abc=[]
    i=1
    count=0
    while True:
        s.listen(0)
        c, addr = s.accept()
        print(addr,"사용자가 접속함")
        data = c.recv(1024)
        c.send(msg)
        decoding(c.recv(1024))
        print(data)
        abc.append(data)
        count+=1
        c.close()
        if count==10:
            file_create_loop(abc)
            i+=1
            abc=[]


def clash_situation():
    global temp
    temp='2'



def scream_situation():
    global temp
    temp='1'

"""            
def serveropne():
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except socket.error as err :
        print("에러 발생 원인 :  %s"%(err))
    temp='0'
    msg=bytearray(temp,'utf-8')
    HOST='192.168.22.106'
    port=1024
    s.bind((HOST,port))
    print("%d 포트로 연결을 기다리는중"%(port))
    abc=[]
    i=1
    count=0
    while True:
        s.listen(0)
        c, addr = s.accept()
        print(addr,"사용자가 접속함")
        data = c.recv(1024)
        c.send(msg)
        decoding(c.recv(1024))
        print(data)
        abc.append(data)
        count+=1
        c.close()
        if count==10:
            file_create_loop(abc)
            i+=1
            abc=[]
"""