## using this Python file to create a voice recognition 

import time as tm
import datetime as dt


def info():
     syn = "This project is UNDER CONSTUCTION  \n " + tm.ctime() + "\nthis project is about creating \na message reply. \n\n"
     for i in syn:
          print(i,end="")
          tm.sleep(0.008)
          
info()

def printi(syntax):
     for i in syntax:
          print(i,end="")
          tm.sleep(0.016)
          

     
