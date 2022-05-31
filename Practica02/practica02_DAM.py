# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_02_03.py 
# para curso de Comunicaciones Digitales
#******************************************************************
# Programador: G. Laguna
# Fecha: 24 de abril 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 23 de noviembre 2021
# Universidad Automoma Metropolitana 
# Unidad Lerma
#******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def d_mod(a):
  T = 0.00005; #Periodo de muestreo (seg)
  fs = 1/T; #Frecuencia de muestreo (20KHz)
  N = 1000 #Numero de muestras
  t = np.arange(0.0, N*T, T) #N muestras para intervalo total
  f = np.arange(0.0, (N/2)*(fs/N)/1000, (fs/N)/1000) #N/2 muestras para intervalo de Nyquist (KHz)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Transmision:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #Senal con componentes de 100, 200 y 300 Hz:
  g = np.cos(2*np.pi*100*t)+0.1*np.cos(2*np.pi*200*t)+0.4*np.cos(2*np.pi*300*t) #%Senal
  c = 2*np.cos(2*np.pi*1000*t) #Portadora de 1KHz
  s = g*c #Senal AM
  S = np.fft.fft(s)
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Recepcion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  teta=np.pi*a #Angulo de desfase respecto a la portadora
  print("rad: ", teta)
  print("deg: ", (teta*180)/np.pi)
  lo = np.cos(2*np.pi*1000*t+teta) #Oscilador local
  y = s*lo #Senal demodulada sin filtrar
  Y = np.fft.fft(y)

  # Filtro LP con corte en 1000 Hz (banda base):
  #b_lp con 32 Coeficientes para el filtro FIR:
  b_lp = signal.firwin(32, 0.1) #Taps, Cutoff = 2*1KHz/20KHz

  #Senal demodulada filtrada:
  u = signal.lfilter(b_lp, 1, y)
  U = np.fft.fft(u)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Graficacion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fig1, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, s)
  axarr[0].set_title('Senal AM s(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(S[0:100]))
  axarr[1].set_title('|S(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig1.subplots_adjust(hspace=0.5)

  fig2, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, y)
  axarr[0].set_title('Senal y(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:200],abs(Y[0:200]))
  axarr[1].set_title('|Y(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig2.subplots_adjust(hspace=0.5)

  fig3, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, u)
  axarr[0].set_title('Senal demodulada  u(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(U[0:100]))
  axarr[1].set_title('|U(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig3.subplots_adjust(hspace=0.5)

  plt.show()

d_mod(1/2)