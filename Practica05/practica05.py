# -*- coding: utf-8 -*-

# ******************************************************************
# Codigo demostrativo Code_07_01.py
# para curso de Comunicaciones Digitales
# ******************************************************************
# Programador: G. Laguna
# Fecha: 29 de enero 2020
# Contribuciones: Adriana de la Cruz Peralta
# 13 de diciembre 2021
# Universidad Automoma Metropolitana
# Unidad Lerma
# ******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def miPLL(x, fo, fs, K0, K1, K2, K3):
    """
  x: secuencia real de entrada
  fo: frecuencia nominal del PLL
  fs: frecuencia de muestreo de las secuencias
  K0: Sensibilidad o ganancia del CNO
  K1,K2,K3: Factores de las ramas proporcionales del filtro de lazo
  Regresa:
  y: secuencia con la evolucion del argumento
     para una funcion sin() o cos()
    """
    Ts = 1.0 /float(fs)  # Periodo de muestreo
    To = 1.0 /float(fo)  # Periodo de la frecuencia nominal
    uq = 2 * np.pi * (Ts / To)  # Paso con la aportacion de la frecuencia nominal

    # Filtro pasa bajas del bloque detector de fase:
    Norder = 5
    b_lp, a_lp = signal.butter(Norder, 2 * (fo / 2.0) / float(fs))
    fstate = np.zeros(Norder)  # Vector con el estado inicial para el filtro

    teta_acc = uq  # Inicializacion del acumulador 1, en el NCO
    acc2 = 0  # Inicializacion del acumulador 2, en el filtro de lazo
    acc3 = 0  # Inicializacion del acumulador 3, en el filtro de lazo

    y = np.zeros(len(x)) #Inicializacion de vector de salida
    m = np.zeros(2)  # Inicializacion de vector de trabajo
    for i in range(len(x)):
        s = -np.sin(teta_acc)  # Senal generada localmente
        # Detector de fase:
        m[0] = s * x[i]
        teta_e, fstate = signal.lfilter(b_lp, a_lp, np.array([m[0]]), zi=fstate)
        # Filtro de lazo:
        acc3 = acc3 + teta_e * K3
        acc2 = acc2 + teta_e * K2 + acc3
        v = teta_e * K1 + acc2
        # CNO:
        teta_acc = teta_acc + v * K0 + uq
        # salida
        y[i] = np.mod(teta_acc, 2 * np.pi)

    return y


def PLL_loop_filter(k1, k2, k3):

    T = 0.00005;  # Periodo de muestreo (seg)
    fs = 1 / T;  # Frecuencia de muestreo (Hz)
    N = 1000  # Numero de muestras
    t = np.arange(0.0, N * T, T)  # N muestras para intervalo total
    f = np.arange(0.0, (N / 2) * (fs / N) / 1000, (fs / N) / 1000)  # N/2 muestras para intervalo de Nyquist (KHz)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Receptor:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Senal de entrada:
    fp = 1000  # Frecuencia de la senal piloto
    fi = np.pi  # Angulo de desfase respecto al oscilador local
    x = np.cos(2 * np.pi * fp * t + fi)  # Senal no sincronizada
    # PLL:
    teta = miPLL(x, fp, fs, 1, k1, k2, k3)  # PLL del tipo 1 (K2=0, K3=0)
    # Oscilador local sincronizado (NCO)
    c = np.cos(teta)

    # Error:
    e = x - c
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Graficacion:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig1, axarr = plt.subplots(3, 1)
    axarr[0].plot(t[0:500], x[0:500])
    axarr[0].set_title('Senal piloto recibida x(t)')
    axarr[0].set_xlabel('Tiempo [s]')
    axarr[0].set_ylabel('Amplitud')
    axarr[1].plot(t[0:500], c[0:500])
    axarr[1].set_title('Senal sincronizada c(t)')
    axarr[1].set_xlabel('Tiempo [s]')
    axarr[1].set_ylabel('Amplitud')
    axarr[2].plot(t[0:500], e[0:500])
    axarr[2].set_title('Error e(t)')
    axarr[2].set_xlabel('Tiempo [s]')
    axarr[2].set_ylabel('Amplitud')
    fig1.subplots_adjust(hspace=1.0)

    plt.show()

PLL_loop_filter(0.1, 0.1, 0.1)