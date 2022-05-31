# -*- coding: utf-8 -*-

#******************************************************************
# Codigo demostrativo Code_01_03.py 
# para curso de Comunicaciones Digitales
#******************************************************************
# Programador: G. Laguna
# Fecha: 24 de abril 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 15 de noviembre 2021
# Universidad Automoma Metropolitana 
# Unidad Lerma
#******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math

def sig_comp(a, b): 
  t = np.arange(0.0, 0.05, 0.05/100 )
  s1 = a*np.cos(2*np.pi*100*t)+b*np.sin(2*np.pi*100*t)

  plt.figure(1)
  plt.plot(t, s1, '-')
  plt.title('Senal')
  plt.xlabel('Tiempo [s]')
  plt.ylabel('Amplitud')

  S1=np.fft.fft(s1)
  f=np.arange(0.0, 1.0 , 1.0/50)
  plt.figure(2)
  plt.stem(f,abs(S1[0:50]))
  plt.xlabel('Frecuencia normalizada a f_s/2')
  plt.ylabel('Magnitud')

  fase = np.angle(S1)
  y= [fase[5]]
  x=[0.1]
  plt.figure(3)
  plt.stem(x,y)
  plt.xlabel('Frecuencia normalizada a f_s/2')
  plt.ylabel('Fase [rad]')
  plt.show()
  
  teta_rad = fase[5]
  teta_deg = math.degrees(teta_rad)
  print(teta_rad)
  print(teta_deg)

  co = 10*np.sin(abs(teta_rad))
  ca = 10*np.cos(abs(teta_rad))
  
  fig = plt.figure(4)
  ax = fig.add_subplot(1, 1, 1)
  ax.spines['left'].set_position('center')
  ax.spines['bottom'].set_position('center')
  plt.quiver(0, 0, a, 0, scale_units='xy', angles='xy', scale=1, color='blue')
  plt.quiver(0, 0, 0, b, scale_units='xy', angles='xy', scale=1, color='green')
  plt.quiver(0, 0, ca, co, scale_units='xy', angles='xy', scale=1, color='red')
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  plt.show()

sig_comp(1, 9)