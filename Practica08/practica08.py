#!/usr/bin/env python
# coding: utf-8

# ******************************************************************
# Codigo demostrativo Code_11_04.py
# para curso de Comunicaciones Digitales
# ******************************************************************
# Programador: G. Laguna
# Fecha: 7 de octubre 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 17 de enero 2022
# Universidad Automoma Metropolitana
# Unidad Lerma
# ******************************************************************

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def upsample(x, L):
    """
    Upsample by factor L

    Insert L - 1 zero samples in between each input sample.

    Parameters
    ----------
    x : ndarray of input signal values
    L : upsample factor

    Returns
    -------
    y : ndarray of the output signal values

    Examples
    --------
    >>> y = upsample(x,3)
    --------
    This code is part of the module ssd.py that was developed to accompany the book
    Signals and Systems for Dummies published by Wiley Publishing.
    Copyright 2012, 2013 Mark Wickert, mwickert@uccs.edu
    v 1.0
    """
    N_input = len(x)
    y = np.hstack((x.reshape(N_input, 1), np.zeros((N_input, L - 1))))
    y = y.flatten()
    return y


def downsample(x, M, p=0):
    """
    Downsample by factor M

    Keep every Mth sample of the input. The phase of the input samples
    kept can be selected.

    Parameters
    ----------
    x : ndarray of input signal values
    M : upsample factor
    p : phase of decimated value, 0 (default), 1, ..., M-1

    Returns
    -------
    y : ndarray of the output signal values

    Examples
    --------
    >>> y = downsample(x,3)
    >>> y = downsample(x,3,1)
    --------
    This code is part of the module ssd.py that was developed to accompany the book
    Signals and Systems for Dummies published by Wiley Publishing.
    Copyright 2012, 2013 Mark Wickert, mwickert@uccs.edu
    v 1.0
    """
    k = int(np.floor((len(x) / M) * M))
    x = x[0:k]
    x = x.reshape((len(x) / M, M))
    y = x[:, p]
    return y


def rand_normal(m,desv,N):
    """
   rand_normal(m,desv,N) genera una variable aleatoria con distribucion normal
  m: media
  desv: desviacion estandar
  N: numero de muestras
    Regresa:
  C: valor de las muestras aleatorias
    """
    C = np.zeros(N)
    for i in xrange(N):
        A = np.random.rand(1)
        R = desv * np.sqrt(2 * np.log(1 / (1 - A)))
        T = np.random.rand(1) * 2 * np.pi
        C[i] = m+R*np.cos(T)

    return C

def bin_rand_uniform(N):
    """
   bin_rand_uniform(N) genera una secuencia binaria con distribucion uniforme
  N: numero de muestras
    Regresa:
  s: secuencia de bits con distribucion uniforme
    """
    s = np.zeros(N)
    for i in xrange(N):
        A = np.random.rand(1)
        if A > 0.5 :
            s[i] = 1
        else:
            s[i] = 0

    return s

def sqrt_Hrc(f,W,Wo):
    """
  sqrt_Hrc(f,W,Wo) para calcular la el espectro de raiz de coseno alzado
  f:     frecuencia en Hertz
  W:     ancho de banda del filtro
  Wo:    ancho de banda de Nyquist
Regresa:
  y: valor resultante
    """
    if abs(f)>W:
        y=0
    elif abs(f)>(2*Wo-W):
        y = np.cos((np.pi / 4) * ((abs(f) + W - 2 * Wo) / (W - Wo)));
    else:
        y=1
    return y

def sample_sqrtHrc(f,W,Wo):
    """
   sample_sqrtHrc(f,W,Wo) para generar una secuencia con la respuesta en frecuencia
                          para un filtro raiz de coseno-alzado
    f:     secuencia de frecuencias en Hertz
    W:     ancho de banda del filtro
    Wo:    ancho de banda de Nyquist
    Regresa:
    s: secuencia resultante
    """
    s = np.zeros(len(f))
    for i in xrange(len(f)):
        s[i] = sqrt_Hrc(f[i],W,Wo)

    return s

def miSimPe_4_QAM_wFormated_pulses_sys(snr_dB, N):
    """
    miSimPe_4_QAM_wFormated_pulses_sys(snr_dB, N) simula comunicacion de un sistema 4-QAM
        con formateo de pulsos y estimacion de la probabilidad de error.
    snr_dB: la relacion SNR en dB
    N: numero de muestras para la simulacion Monte Carlo
    Regresa:
    p: la estimacion de la probabilidad de error
    """

    #Catalogo de simbolos como proyecciones [I,Q]:
    s0 = np.array([-1.0, -1.0])
    s1 = np.array([1.0, -1.0])
    s2 = np.array([1.0, 1.0])
    s3 = np.array([-1.0, 1.0])

    #Potencia del ruido:
    E = 2 #Energia de las senales 4-arias
    snr = 10 ** (snr_dB / float(10))
    sigma = np.sqrt(E/snr)

    #Ruido aditivo:
    ni = rand_normal(0, sigma, N)
    nq = rand_normal(0, sigma, N)

    #Generacion de datos binarios:
    s = bin_rand_uniform(2*N)

    #******************************************************************
    #Transmision 4-arias:
    #******************************************************************
    ti = np.zeros(N)
    tq = np.zeros(N)
    for i in xrange(N):
        #Mapeo:
        if s[i] == 0 and s[N+i] == 0:
            ti[i] = s0[0]
            tq[i] = s0[1]
        elif s[i] == 0 and s[N+i] == 1:
            ti[i] = s1[0]
            tq[i] = s1[1]
        elif s[i] == 1 and s[N+i] == 1:
            ti[i] = s2[0]
            tq[i] = s2[1]
        elif s[i] == 1 and s[N+i] == 0:
            ti[i] = s3[0]
            tq[i] = s3[1]
        

    #Upsample xFactor:
    UpDownFactor = 10
    tix10 = upsample(ti, UpDownFactor)
    tqx10 = upsample(tq, UpDownFactor)

    #Formateo de pulsos:
    T = 1 #Se normaliza respecto de la frecuencia de muestreo
    Wo = 1 / float(2 * T) #Frecuencia de Nyquist
    taps = 65 #Numero de coeficientes del filtro
    K = taps / 2
    step = 1 / float(K * T)
    f = np.arange(-1 , 1 , step)
    alfa = 0.5 #Factor de despliegue
    W = alfa * Wo + Wo
    H = sample_sqrtHrc(f, W, Wo)
    hc = np.fft.ifft(H)
    h = np.real(hc)
    L=len(h)
    b = np.concatenate((h[L-K:L], h[0:K]), axis=None)
    #Filtro de raiz de coseno alzado:
    tix10z = np.concatenate((tix10, np.zeros(taps)) , axis=None)
    tqx10z = np.concatenate((tqx10, np.zeros(taps)) , axis=None)
    ti_pulses = signal.lfilter(b, 1, tix10z)
    tq_pulses = signal.lfilter(b, 1, tqx10z)

    #******************************************************************
    #Recepcion:
    #******************************************************************
    #Filtro de raiz de coseno alzado:
    ri_pulses =signal.lfilter(b, 1, ti_pulses)
    rq_pulses =signal.lfilter(b, 1, tq_pulses)

    #Control automatico de ganancia
    max_src = max(abs(tix10z))
    max_pulses = max(abs(ri_pulses))
    G = max_src / max_pulses #Ganancia
    ri_eq = ri_pulses * G
    rq_eq = rq_pulses * G

    #Sincronizacion:
    size = len(ri_eq)
    ri_sync = ri_eq[taps-1 : size]
    rq_sync = rq_eq[taps-1 : size]

    fase = 0
    rid10 = downsample(ri_sync, UpDownFactor, fase)
    rqd10 = downsample(rq_sync, UpDownFactor, fase)

    #Potencia del ruido:
    E = 2 #Energia de las senales 4-arias
    snr = 10 ** (snr_dB / float(10))
    sigma = np.sqrt(E/snr)

    #Ruido aditivo:
    ni = rand_normal(0, sigma, N)
    nq = rand_normal(0, sigma, N)
    ripn = rid10[0:N]+ni
    rqpn = rqd10[0:N]+nq

    #Deteccion 4-aria y conteo de errores:
    errores = 0

    for i in xrange(N):
        #Mapeo y dato original:
        if s[i] == 0 and s[N+i] == 0:
            sm = [0, 0]
        elif s[i] == 0 and s[N+i] == 1:
            sm = [0, 1]
        elif s[i] == 1 and s[N+i] == 1:
            sm = [1, 1]
        elif s[i] == 1 and s[N+i] == 0:
            sm = [1, 0]
        

        #A la salida del matched filter:
        r = np.array([ripn[i] , rqpn[i]])

        #Detector:
        d_min = np.linalg.norm(s0 - r) #Distancia minima de entrada
        d = [0, 0]  #Datos detectados por defecto

        if np.linalg.norm(s1 - r) < d_min :
            d_min = np.linalg.norm(s1 - r)
            d = [0, 1]

        if np.linalg.norm(s2 - r) < d_min :
            d_min = np.linalg.norm(s2 - r)
            d = [1, 1]

        if np.linalg.norm(s3 - r) < d_min :
            d_min = np.linalg.norm(s3 - r)
            d = [1, 0]
        

        #Conteo de errores:
        if d[0]!=sm[0]:
            errores = errores + 1

        if d[1] != sm[1]:
            errores = errores + 1


    #Estimacion de la probabilidad de error:
    p = errores / float(2*N)
    return p

if __name__ == '__main__':
    # Test bench area
    #print("Hola mundo")

    N=10000 #Numero de experimentos de la simulacion Monte Carlo
    snr_in_dB = np.arange(0.0, 25.0, 1.0)

    Pe = np.zeros(len(snr_in_dB))
    for i in xrange(len(snr_in_dB)):
        Pe[i]=miSimPe_4_QAM_wFormated_pulses_sys(snr_in_dB[i], N)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Graficacion:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.semilogy(snr_in_dB,Pe)
    plt.title('Pe para deteccion 4-QAM con simulacion Monte Carlo')
    plt.xlabel('SNR en dB')

    plt.show()