# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:20:21 2021

@autores: Ezequiel, Franco, y Luna
"""

import numpy as np
from matplotlib import pyplot as plt
import librosa
from scipy import signal as sgn
import sounddevice as sd

fs1 = 44100

x,Fs = librosa.load('sinesweep.wav')

 
f1,t1,x_mag = sgn.stft(x,Fs,window = 'blackman', nperseg = 1000)

iteracion = 20
 
M = abs(x_mag)

P = np.random.randn(*M.shape)

for i in range(iteracion):
            
    D = M * np.exp(1j * P)
    aproximacion = librosa.istft(D)    
    
    
fs_aprox = Fs/2

f2,t2,x2 = sgn.stft(aproximacion,fs_aprox,window = 'blackman', nperseg = 500)

x = x/(abs(max(x)))
aproximacion = aproximacion/(abs(max(aproximacion)))

plt.figure()
plt.pcolormesh(t1,f1,20*np.log10(abs(x_mag**2)))

plt.figure()
plt.pcolormesh(t2,f2,20*np.log10(abs(x2**2)))

plt.figure()
plt.pcolormesh(t1,f1,np.unwrap(np.angle(x_mag**2)))

plt.figure()
plt.pcolormesh(t2,f2,np.unwrap(np.angle(x2**2)))

#x = 0.4 * x[0:10*Fs]
aproximacion = 0.3*aproximacion

sd.play(aproximacion,fs_aprox)




