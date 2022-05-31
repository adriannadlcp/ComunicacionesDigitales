# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_05_01.py
# para curso de Comunicaciones Digitales
#******************************************************************
# Programador: G. Laguna
# Fecha: 1 de mayo 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 7 de diciembre 2021
# Universidad Automoma Metropolitana
# Unidad Lerma
#******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def mod_fm(i):

  T = 0.00005; #Periodo de muestreo (seg)
  fs = 1/T; #Frecuencia de muestreo (Hz)
  N = 1000 #Numero de muestras
  t = np.arange(0.0, N*T, T) #N muestras para intervalo total
  f = np.arange(0.0, (N/2)*(fs/N)/1000, (fs/N)/1000) #N/2 muestras para intervalo de Nyquist (KHz)
  #Senal con componente cosenoidal unica:
  f_i = 300 #frecuencia de la informacion
  g = np.cos(2*np.pi*f_i*t) #%Senal
  G = np.fft.fft(g)
  #Portadora:
  A_c = 1.0 # Amplitud de la portadora
  f_c = 1000 # frecuencia de la portadora
  c = A_c*np.cos(2*np.pi*f_c*t) #Portadora
  C = np.fft.fft(c)
  #Modulacion FM:
  B_fm = i #Indice de modulacion
  s = A_c*np.cos(2*np.pi*f_c*t+ B_fm*np.sin(2*np.pi*f_i*t)) #Senal FM
  S = np.fft.fft(s)

  fig1, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, g)
  axarr[0].set_title('Senal g(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(G[0:100]))
  axarr[1].set_title('|G(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig1.subplots_adjust(hspace=0.5)

  fig2, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, c)
  axarr[0].set_title('Portadora c(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(C[0:100]))
  axarr[1].set_title('|C(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig2.subplots_adjust(hspace=0.5)

  fig3, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, s)
  axarr[0].set_title('Senal FM s(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(S[0:100]))
  axarr[1].set_title('|S(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig3.subplots_adjust(hspace=0.5)

  plt.show()

mod_fm(1.5)