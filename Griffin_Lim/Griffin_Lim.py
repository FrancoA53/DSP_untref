# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:55:30 2021

@author: ACER
"""


import numpy as np
import librosa as lb
from scipy import signal as sgn


def griffin_lim(y, fs, iteracion, ventana = 'hann'):
   
    y_real = np.real(y)
    matriz_compleja = np.random.randn(*y_real.shape)
    matriz_reconstruida = y_real* e** (1j*matriz_compleja)
    b, ISTFT_inicial = sgn.istft(matriz_reconstruida, fs, window = ventana, nperseg = 1000)
    
    counter= 0
    
    while counter<iteracion:
    
        f2,t2,inicio = sgn.stft(ISTFT_inicial,fs, window = ventana , nperseg = 1000)
        inicio_compleja = np.imag(inicio)
        E = y_real * np.exp(1j*inicio_compleja)
        t,ISTFT_initial = sgn.istft(E,fs,window = ventana , nperseg = 1000)
        counter += 1
    
    return t,ISTFT_initial

def griffin_lim2(x,fs,iteracion,ventana = 'hann'):
    
    x_mag = lb.stft(x,fs,window = ventana, nperseg = 1000)
     
    M = abs(x_mag)
    
    P = np.random.randn(*M.shape)
    
    for i in range(iteracion):
                
        D = M * np.exp(1j * P)
        aproximacion = lb.istft(D)    
        
    return aproximacion
