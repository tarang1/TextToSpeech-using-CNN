# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 00:49:59 2018

@author: Tarang
"""
import os
from playsound import playsound
from gtts import gTTS
import random
r1 = random.randint(1,10000000)
r2 = random.randint(1,10000000)

randfile = str(r2)+"randomtext"+str(r1) +".mp3"
language = 'en'
file = open("text.txt","r+")
te = file.read()
file.close()
obj = gTTS(text=te,lang=language,slow=True)
obj.save(randfile)
#os.system(randfile)
playsound(randfile)
os.remove(randfile)

