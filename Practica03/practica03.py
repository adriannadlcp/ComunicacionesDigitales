# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_04_07.py
# para curso de Comunicaciones Digitales
#******************************************************************
# Programador: G. Laguna
# Fecha: 25 de abril 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 29 de noviembre 2021
# Universidad Automoma Metropolitana
# Unidad Lerma
#******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def saturate(x):
    """
    function y = saturate(x)
    where x is a real sequence.

    Gerardo Laguna
    """
    y = np.zeros(len(x))  # y is the result sequence

    for i in range(len(x)):
        if x[i]>0:
            y[i]=x[i]

    return y

def d_mod_com(a):
  T = 0.00005; #Periodo de muestreo (seg)
  fs = 1/T; #Frecuencia de muestreo (Hz)
  N = 1000 #Numero de muestras
  t = np.arange(0.0, N*T, T) #N muestras para intervalo total
  f = np.arange(0.0, (N/2)*(fs/N)/1000, (fs/N)/1000) #N/2 muestras para intervalo de Nyquist (KHz)
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Transmision:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #Offset:
  A=1.0
  #Senal con componentes de 100, 200 y 300 Hz:
  g = A+np.cos(2*np.pi*100*t)+0.1*np.cos(2*np.pi*200*t)+0.4*np.cos(2*np.pi*300*t) #%Senal
  c = np.cos(2*np.pi*1000*t) #Portadora de 1KHz
  # Senal AM-DSB-TC:
  s = g*c
  #Transmision:
  c_tx = np.cos(2*np.pi*8000*t) #Portadora para transmision de 8KHz
  s_tx = s*c_tx

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Recepcion en cuadratura:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Oscilador local de 8KHz:
  teta = np.pi * a  # Angulo de desfase respecto a la portadora
  lo_i = np.cos(2 * np.pi * 8000 * t + teta)  # Oscilador local en fase
  lo_q = -np.sin(2 * np.pi * 8000 * t + teta)  # Oscilador local en cuadratura
  # Canales demodulados sin filtrar:
  y_re = s_tx * lo_i
  y_im = s_tx * lo_q

  # Filtro LP con corte en 2000 Hz (banda base AM):
  #b_lp con 64 Coeficientes para el filtro FIR:
  b_lp = signal.firwin(64, 0.2) #Taps, Cutoff = 2*2KHz/20KHz

  # Canales demodulados filtrados:
  x_re = signal.lfilter(b_lp, 1, y_re)
  X_re = np.fft.fft(x_re)
  x_im = signal.lfilter(b_lp, 1, y_im)
  X_im = np.fft.fft(x_im)

  #Senal detectada en forma compleja:
  y_s=saturate(x_re+1j*x_im)

  # Filtro LP con corte en 300 Hz (banda base original):
  #b_lp2 con 128 Coeficientes para el filtro FIR:
  b_lp2 = signal.firwin(128, 0.03) #Taps, Cutoff = 2*0.03KHz/20KHz

  #Senal detectada:
  u = signal.lfilter(b_lp2, 1, y_s)
  U = np.fft.fft(u)
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Graficacion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fig1, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, x_re)
  axarr[0].set_title('Senal AM-DSB-TC  x_re(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:500],abs(X_re[0:500]))
  axarr[1].set_title('|X_re(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig1.subplots_adjust(hspace=0.5)

  plt.figure(2)
  plt.plot(t, y_s, '-')
  plt.title('Senal saturada y_s(t)')
  plt.xlabel('Tiempo [s]')
  plt.ylabel('Amplitud')

  fig3, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, u)
  axarr[0].set_title('Senal detectada  u(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:100],abs(U[0:100]))
  axarr[1].set_title('|U(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig3.subplots_adjust(hspace=0.5)

  plt.show()

d_mod_com(1/8)