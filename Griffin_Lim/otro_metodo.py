# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 19:39:45 2021

@author: Franco
"""

import numpy as np
from matplotlib import pyplot as plt
import librosa as lb
from scipy import signal as sgn
import sounddevice as sd

fs1 = 44100

x,Fs = lb.load('sinesweep.wav')

 
f1,t1,y = sgn.stft(x,Fs, nperseg = 1000)
M = np.real(y)
P = np.random.randn(*M.shape)
D= M + 1j*P
b, ISTFT_initial = sgn.istft(D)

counter= 0
iteracion = 20

while counter<iteracion:

    
    f2,t2,inicio = sgn.stft(ISTFT_initial,Fs, nperseg = 1000)
    imag = np.imag(inicio)
    E = M * np.exp(1j*imag)
    t3,ISTFT_initial = sgn.istft(E)
    counter += 1


freq,tiemp, Zxx = sgn.stft(ISTFT_initial,Fs, nperseg = 1000)



x = x/(max(abs(x)))

aproximacion = ISTFT_initial/max(abs(ISTFT_initial))
    

plt.figure()
plt.pcolormesh(t1,f1,20*np.log10(abs(y**2)))

plt.figure()
plt.pcolormesh(tiemp,freq,20*np.log10(abs(Zxx**2)))
plt.xlabel('lauti putito')
plt.show()

# plt.figure()
# plt.pcolormesh(t1,f1,np.unwrap(np.angle(y**2)))

# plt.figure()
# plt.pcolormesh(tiemp,freq,np.unwrap(np.angle(Zxx**2)))


# x = 0.5*x

# sd.play(x,Fs)

# proximacion = 0.3*aproximacion

# sd.play(aproximacion,Fs)