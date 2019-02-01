
# coding: utf-8

# In[ ]:


import socket

def openSoket():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.103.63', 1024))
    while 1>0:
        server_socket.listen(0)
        client_socket, addr = server_socket.accept()
        data = client_socket.recv(65535)
        print(data)

