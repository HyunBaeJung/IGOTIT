# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:52:37 2019

@author: gusqo
"""

import socket

def decoding(a):
    a=a
    b=a.decode("utf-8")
    print(b)
try:
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print("소켓 생성완료")
except socket.error as err :
    print("에러 발생 원인 :  %s"%(err))
 
temp="it's ok man"
msg=bytearray(temp,'utf-8')
HOST='192.168.43.246'
port=1024
s.bind((HOST,port))
print("%d 포트로 연결을 기다리는중"%(port))

while True:
    s.listen(0)
    c, addr = s.accept()
    print(addr,"사용자가 접속함")
    data = c.recv(1024)
    c.send(msg)
    #decoding(c.recv(1024))
    print(data)
    c.close()
    
    