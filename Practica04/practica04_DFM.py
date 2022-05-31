# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_05_06.py
# para curso de Comunicaciones Digitales
#******************************************************************
# Programador: G. Laguna
# Fecha: 6 de mayo 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 7 de diciembre 2021
# Universidad Automoma Metropolitana
# Unidad Lerma
#******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def cmplx_discriminator(x,y):
    """
  x: secuencia de entrada con la parte real
  y: secuencia de entrada con la parte imaginaria
  Regresa:
  s: secuencia discriminada
    """
    # Coeficientes para filtro diferenciador:
    b = [1, -1]
    # Derivacion:
    dx = signal.lfilter(b, 1, x) #Diferencial
    dy = signal.lfilter(b, 1, y) #Diferencial
    #Manipulaciones:
    s_a = dy * x - dx * y;
    s_b = x * x + y * y;
    s = s_a / s_b;

    return s

def dmod_fm(i):
  T = 0.00005; #Periodo de muestreo (seg)
  fs = 1/T; #Frecuencia de muestreo (Hz)
  N = 1000 #Numero de muestras
  t = np.arange(0.0, N*T, T) #N muestras para intervalo total
  f = np.arange(0.0, (N/2)*(fs/N)/1000, (fs/N)/1000) #N/2 muestras para intervalo de Nyquist (KHz)
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Transmision:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  #Senal con componentes de 100, 200 y 300 Hz:
  g = np.cos(2*np.pi*100*t)+0.1*np.cos(2*np.pi*200*t)+0.4*np.cos(2*np.pi*300*t) #%Senal
  #Integracion de senal:
  teta=np.imag(signal.hilbert(g))
  #Portadora:
  A_c = 1.0 # Amplitud de la portadora
  f_c = 1000 # frecuencia de la portadora
  #Modulacion FM:
  B_fm= 2.0 # Indice de modulacion
  s = A_c*np.cos(2*np.pi*f_c*t+B_fm*teta) #Senal FM
  S = np.fft.fft(s)

  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Recepcion en cuadratura:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Oscilador local:
  fi = i  # Angulo de desfase respecto a la portadora
  lo_i = np.cos(2 * np.pi * f_c * t + fi)  # Oscilador local en fase
  lo_q = -np.sin(2 * np.pi * f_c * t + fi)  # Oscilador local en cuadratura
  # Canales mezclados sin filtrar:
  x_re = s * lo_i
  x_im = s * lo_q

  # Filtro LP con corte en 500 Hz (banda base):
  #b_lp con 128 Coeficientes para el filtro FIR:
  b_lp = signal.firwin(128, 0.05) #Taps, Cutoff = 2*0.05KHz/20KHz

  # Canales mezclados y filtrados:
  y_re = signal.lfilter(b_lp, 1, x_re)
  y_im = signal.lfilter(b_lp, 1, x_im)

  #Demodulacion por discriminador complejo:
  z = cmplx_discriminator(y_re, y_im)
  Z = np.fft.fft(z)


  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  # Graficacion:
  # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  fig1, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, s)
  axarr[0].set_title('Senal FM recibida s(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:500],abs(S[0:500]))
  axarr[1].set_title('|S(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig1.subplots_adjust(hspace=0.5)

  fig2, axarr = plt.subplots(2, 1)
  axarr[0].plot(t, z)
  axarr[0].set_title('Senal discriminada  z(t)')
  axarr[0].set_xlabel('Tiempo [s]')
  axarr[1].stem(f[0:500],abs(Z[0:500]))
  axarr[1].set_title('|Z(f)|')
  axarr[1].set_xlabel('Frecuencia [KHz]')
  fig2.subplots_adjust(hspace=0.5)

  plt.show()

dmod_fm(3)