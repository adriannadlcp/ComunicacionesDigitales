# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_02_06.py 
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


def d_mod_q(a):
  T = 0.00005; #Periodo de muestreo (seg)
  fs = 1/T; #Frecuencia de muestreo (20KHz)
  N = 1000 #Numero de muestras
  t = np.arange(0.0, N*T, T) #N muestras para intervalo total
  f = np.arange(0.0, (N/2)*(fs/N)/1000, (fs/N)/1000) #N/2 muestras para intervalo de Nyquist (KHz)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Transmision:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #Senal 1 con componentes de 100, 200 y 300 Hz:
  g1 = np.cos(2*np.pi*100*t)+0.1*np.sin(2*np.pi*200*t)+0.4*np.cos(2*np.pi*300*t) #%Senal 1
  #Senal 2 con componentes de 100, 200 y 300 Hz:
  g2 = 0.4*np.sin(2*np.pi*100*t)+0.1*np.cos(2*np.pi*200*t)+np.sin(2*np.pi*300*t) #%Senal 2
  #Modulacion en cuadratura con portadoras de 1KHz:
  s = g1*np.cos(2*np.pi*1000*t)-g2*np.sin(2*np.pi*1000*t) #Senal QAM
  S = np.fft.fft(s)
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Recepcion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #Oscilador local de 1KHz:
  teta=np.pi*a #Angulo de desfase respecto a la portadora
  print("rad: ", teta)
  print("deg: ", (teta*180)/np.pi)
  lo_i = np.cos(2 * np.pi * 1000 * t+teta)  # Oscilador local en fase
  lo_q = -np.sin(2 * np.pi * 1000 * t+teta)  # Oscilador local en cuadratura
  #Canales demodulados sin filtrar:
  y1 = s*lo_i 
  y2 = s*lo_q 
      
  # Filtro LP con corte en 1000 Hz (banda base):
  #b_lp con 32 Coeficientes para el filtro FIR:
  b_lp = signal.firwin(32, 0.1) #Taps, Cutoff = 2*1KHz/20KHz

  #Canales demodulados filtrados:
  z1 = signal.lfilter(b_lp, 1, y1)
  Z1 = np.fft.fft(z1)
  z2 = signal.lfilter(b_lp, 1, y2)
  Z2 = np.fft.fft(z2)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Graficacion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fig1, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, s)
  axarr[0].set_title('Senal QAM s(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(S[0:100]))
  axarr[1].set_title('|S(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig1.subplots_adjust(hspace=0.5)

  fig2, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, z1)
  axarr[0].set_title('Senal demodulada  z1(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(Z1[0:100]))
  axarr[1].set_title('|Z1(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig2.subplots_adjust(hspace=0.5)

  fig3, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, z2)
  axarr[0].set_title('Senal demodulada  z2(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(Z2[0:100]))
  axarr[1].set_title('|Z2(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig3.subplots_adjust(hspace=0.5)

  plt.show()

d_mod_q(1/2)