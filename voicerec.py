## using this Python file to create a voice recognition 

import time as tm


def info():
     syn = "This project is UNDER CONSTUCTION  \n " + tm.ctime() + "\nthis project is about creating \na voice recognition program. \n\n"
     for i in syn:
          print(i,end="")
          tm.sleep(0.005)
          
info()
