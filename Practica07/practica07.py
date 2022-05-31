# -*- coding: utf-8 -*-


# ******************************************************************
# Codigo demostrativo Code_10_02.py
# para curso de Comunicaciones Digitales
# ******************************************************************
# Programador: G. Laguna
# Fecha: 24 de junio 2019
# Contribuciones: Adriana de la Cruz Peralta
# Fecha: 11 de enero 2022
# Universidad Automoma Metropolitana
# Unidad Lerma
# ******************************************************************


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


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
    for i in range(N):
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
    for i in range(N):
        A = np.random.rand(1)
        if A > 0.5 :
            s[i] = 1
        else:
            s[i] = 0

    return s


def miSimPe_64_QAM_sys(snr_dB, N):
    """
    miSimPe_64_QAM_sys(snr_dB, N) simula comunicacion de un sistema  64-QAM y estimar
                                  la probabilidad de error.
    snr_dB: la relacion SNR en dB
    N: numero de muestras para la simulacion Monte Carlo
    Regresa:
    p: la estimacion de la probabilidad de error
    """

    #Catalogo de simbolos como proyecciones [I,Q]:
    s0 = np.array([-7.0, 7.0])
    s1 = np.array([-5.0, 7.0])
    s2 = np.array([-3.0, 7.0])
    s3 = np.array([-1.0, 7.0])
    s4 = np.array([1.0, 7.0])
    s5 = np.array([3.0, 7.0])
    s6 = np.array([5.0, 7.0])
    s7 = np.array([7.0, 7.0])
    s8 = np.array([-7.0, 5.0])
    s9 = np.array([-5.0, 5.0])
    sa = np.array([-3.0, 5.0])
    sb = np.array([-1.0, 5.0])
    sc = np.array([1.0, 5.0])
    sd = np.array([3.0, 5.0])
    se = np.array([5.0, 5.0])
    sf = np.array([7.0, 5.0])
    s10 = np.array([-7.0, 3.0])
    s11 = np.array([-5.0, 3.0])
    s12 = np.array([-3.0, 3.0])
    s13 = np.array([-1.0, 3.0])
    s14 = np.array([1.0, 3.0])
    s15 = np.array([3.0, 3.0])
    s16 = np.array([5.0, 3.0])
    s17 = np.array([7.0, 3.0])
    s18 = np.array([-7.0, 1.0])
    s19 = np.array([-5.0, 1.0])
    s1a = np.array([-3.0, 1.0])
    s1b = np.array([-1.0, 1.0])
    s1c = np.array([1.0, 1.0])
    s1d = np.array([3.0, 1.0])
    s1e = np.array([5.0, 1.0])
    s1f = np.array([7.0, 1.0])
    s20 = np.array([-7.0, -1.0])
    s21 = np.array([-5.0, -1.0])
    s22 = np.array([-3.0, -1.0])
    s23 = np.array([-1.0, -1.0])
    s24 = np.array([1.0, -1.0])
    s25 = np.array([3.0, -1.0])
    s26 = np.array([5.0, -1.0])
    s27 = np.array([7.0, -1.0])
    s28 = np.array([-7.0, -3.0])
    s29 = np.array([-5.0, -3.0])
    s2a = np.array([-3.0, -3.0])
    s2b = np.array([-1.0, -3.0])
    s2c = np.array([1.0, -3.0])
    s2d = np.array([3.0, -3.0])
    s2e = np.array([5.0, -3.0])
    s2f = np.array([7.0, -3.0])
    s30 = np.array([-7.0, -5.0])
    s31 = np.array([-5.0, -5.0])
    s32 = np.array([-3.0, -5.0])
    s33 = np.array([-1.0, -5.0])
    s34 = np.array([1.0, -5.0])
    s35 = np.array([3.0, -5.0])
    s36 = np.array([5.0, -5.0])
    s37 = np.array([7.0, -5.0])
    s38 = np.array([-7.0, -7.0])
    s39 = np.array([-5.0, -7.0])
    s3a = np.array([-3.0, -7.0])
    s3b = np.array([-1.0, -7.0])
    s3c = np.array([1.0, -7.0])
    s3d = np.array([3.0, -7.0])
    s3e = np.array([5.0, -7.0])
    s3f = np.array([7.0, -7.0])

    #Potencia del ruido:
    E = 42 #Energia de las senales 64-arias
    snr = 10 ** (snr_dB / float(10))
    sigma = np.sqrt(E/snr)

    #Generacion de datos binarios:
    s = bin_rand_uniform(8*N)

    #Ruido aditivo:
    ni = rand_normal(0, sigma, N)
    nq = rand_normal(0, sigma, N)

    #Deteccion 16-aria y conteo de errores:
    errores = 0

    for i in range(N):
        # Mapeo y salidas de correlacionadores:
        if s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 0, 0, 0, 0]
            ri = s0[0] + ni[i]
            rq = s0[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 0, 0, 0, 1]
            ri = s1[0] + ni[i]
            rq = s1[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 0, 0, 1, 0]
            ri = s2[0] + ni[i]
            rq = s2[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 0, 0, 1, 1]
            ri = s3[0] + ni[i]
            rq = s3[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 0, 1, 0, 0]
            ri = s4[0] + ni[i]
            rq = s4[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 0, 1, 0, 1]
            ri = s5[0] + ni[i]
            rq = s5[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 0, 1, 1, 0]
            ri = s6[0] + ni[i]
            rq = s6[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 0, 1, 1, 1]
            ri = s7[0] + ni[i]
            rq = s7[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 1, 0, 0, 0]
            ri = s8[0] + ni[i]
            rq = s8[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 1, 0, 0, 1]
            ri = s9[0] + ni[i]
            rq = s9[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 1, 0, 1, 0]
            ri = sa[0] + ni[i]
            rq = sa[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 1, 0, 1, 1]
            ri = sb[0] + ni[i]
            rq = sb[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 1, 1, 0, 0]
            ri = sc[0] + ni[i]
            rq = sc[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 1, 1, 0, 1]
            ri = sd[0] + ni[i]
            rq = sd[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 0, 1, 1, 1, 0]
            ri = se[0] + ni[i]
            rq = se[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 0, 1, 1, 1, 1]
            ri = sf[0] + ni[i]
            rq = sf[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 0, 0, 0, 0]
            ri = s10[0] + ni[i]
            rq = s10[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 0, 0, 0, 1]
            ri = s11[0] + ni[i]
            rq = s11[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 0, 0, 1, 0]
            ri = s12[0] + ni[i]
            rq = s12[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 0, 0, 1, 1]
            ri = s13[0] + ni[i]
            rq = s13[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 0, 1, 0, 0]
            ri = s14[0] + ni[i]
            rq = s14[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 0, 1, 0, 1]
            ri = s15[0] + ni[i]
            rq = s15[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 0, 1, 1, 0]
            ri = s16[0] + ni[i]
            rq = s16[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 0, 1, 1, 1]
            ri = s17[0] + ni[i]
            rq = s17[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 1, 0, 0, 0]
            ri = s18[0] + ni[i]
            rq = s18[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 1, 0, 0, 1]
            ri = s19[0] + ni[i]
            rq = s19[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 1, 0, 1, 0]
            ri = s1a[0] + ni[i]
            rq = s1a[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 1, 0, 1, 1]
            ri = s1b[0] + ni[i]
            rq = s1b[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 1, 1, 0, 0]
            ri = s1c[0] + ni[i]
            rq = s1c[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 1, 1, 0, 1]
            ri = s1d[0] + ni[i]
            rq = s1d[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 0, 1, 1, 1, 1, 0]
            ri = s1e[0] + ni[i]
            rq = s1e[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 0 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 0, 1, 1, 1, 1, 1]
            ri = s1f[0] + ni[i]
            rq = s1f[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 0, 0, 0, 0]
            ri = s20[0] + ni[i]
            rq = s20[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 0, 0, 0, 1]
            ri = s21[0] + ni[i]
            rq = s21[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 0, 0, 1, 0]
            ri = s22[0] + ni[i]
            rq = s22[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 0, 0, 1, 1]
            ri = s23[0] + ni[i]
            rq = s23[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 0, 1, 0, 0]
            ri = s24[0] + ni[i]
            rq = s24[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 0, 1, 0, 1]
            ri = s25[0] + ni[i]
            rq = s25[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 0, 1, 1, 0]
            ri = s26[0] + ni[i]
            rq = s26[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 0, 1, 1, 1]
            ri = s27[0] + ni[i]
            rq = s27[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 1, 0, 0, 0]
            ri = s28[0] + ni[i]
            rq = s28[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 1, 0, 0, 1]
            ri = s29[0] + ni[i]
            rq = s29[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 1, 0, 1, 0]
            ri = s2a[0] + ni[i]
            rq = s2a[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 1, 0, 1, 1]
            ri = s2b[0] + ni[i]
            rq = s2b[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 1, 1, 0, 0]
            ri = s2c[0] + ni[i]
            rq = s2c[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 1, 1, 0, 1]
            ri = s2d[0] + ni[i]
            rq = s2d[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 0, 1, 1, 1, 0]
            ri = s2e[0] + ni[i]
            rq = s2e[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 0 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 0, 1, 1, 1, 1]
            ri = s2f[0] + ni[i]
            rq = s2f[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 0, 0, 0, 0]
            ri = s30[0] + ni[i]
            rq = s30[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 0, 0, 0, 1]
            ri = s31[0] + ni[i]
            rq = s31[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 0, 0, 1, 0]
            ri = s32[0] + ni[i]
            rq = s32[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 0, 0, 1, 1]
            ri = s33[0] + ni[i]
            rq = s33[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 0, 1, 0, 0]
            ri = s34[0] + ni[i]
            rq = s34[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 0, 1, 0, 1]
            ri = s35[0] + ni[i]
            rq = s35[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 0, 1, 1, 0]
            ri = s36[0] + ni[i]
            rq = s36[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 0 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 0, 1, 1, 1]
            ri = s37[0] + ni[i]
            rq = s37[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 1, 0, 0, 0]
            ri = s38[0] + ni[i]
            rq = s38[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 1, 0, 0, 1]
            ri = s39[0] + ni[i]
            rq = s39[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 1, 0, 1, 0]
            ri = s3a[0] + ni[i]
            rq = s3a[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 0 and s[6 * N + i] == 1 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 1, 0, 1, 1]
            ri = s3b[0] + ni[i]
            rq = s3b[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 1, 1, 0, 0]
            ri = s3c[0] + ni[i]
            rq = s3c[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 0 and s[7 * N + i] == 1:
            sm = [0, 0, 1, 1, 1, 1, 0, 1]
            ri = s3d[0] + ni[i]
            rq = s3d[1] + nq[i]
        elif s[i] == 0 and s[N + i] == 0 and s[2 * N + i] == 1 and s[3 * N + i] == 1 and s[4 * N + i] == 1 and \
                s[5 * N + i] == 1 and s[6 * N + i] == 1 and s[7 * N + i] == 0:
            sm = [0, 0, 1, 1, 1, 1, 1, 0]
            ri = s3e[0] + ni[i]
            rq = s3e[1] + nq[i]
        else:
            sm = [0, 0, 1, 1, 1, 1, 1, 1]
            ri = s3f[0] + ni[i]
            rq = s3f[1] + nq[i]

        #Recepcion y salida de correlacionadores:
        r = np.array([ri , rq])

        #Detector:
        d_min = np.linalg.norm(s0 - r) #Distancia minima de entrada
        d = [0, 0, 0, 0, 0, 0, 0, 0]  #Datos detectados por defecto

        if np.linalg.norm(s1 - r) < d_min:
            d_min = np.linalg.norm(s1 - r)
            d = [0, 0, 0, 0, 0, 0, 0, 1]

        if np.linalg.norm(s2 - r) < d_min:
            d_min = np.linalg.norm(s2 - r)
            d = [0, 0, 0, 0, 0, 0, 1, 0]

        if np.linalg.norm(s3 - r) < d_min:
            d_min = np.linalg.norm(s3 - r)
            d = [0, 0, 0, 0, 0, 0, 1, 1]

        if np.linalg.norm(s4 - r) < d_min:
            d_min = np.linalg.norm(s4 - r)
            d = [0, 0, 0, 0, 0, 1, 0, 0]

        if np.linalg.norm(s5 - r) < d_min:
            d_min = np.linalg.norm(s5 - r)
            d = [0, 0, 0, 0, 0, 1, 0, 1]

        if np.linalg.norm(s6 - r) < d_min:
            d_min = np.linalg.norm(s6 - r)
            d = [0, 0, 0, 0, 0, 1, 1, 0]

        if np.linalg.norm(s7 - r) < d_min:
            d_min = np.linalg.norm(s7 - r)
            d = [0, 0, 0, 0, 0, 1, 1, 1]

        if np.linalg.norm(s8 - r) < d_min:
            d_min = np.linalg.norm(s8 - r)
            d = [0, 0, 0, 0, 1, 0, 0, 0]

        if np.linalg.norm(s9 - r) < d_min:
            d_min = np.linalg.norm(s9 - r)
            d = [0, 0, 0, 0, 1, 0, 0, 1]

        if np.linalg.norm(sa - r) < d_min:
            d_min = np.linalg.norm(sa - r)
            d = [0, 0, 0, 0, 1, 0, 1, 0]

        if np.linalg.norm(sb - r) < d_min:
            d_min = np.linalg.norm(sb - r)
            d = [0, 0, 0, 0, 1, 0, 1, 1]

        if np.linalg.norm(sc - r) < d_min:
            d_min = np.linalg.norm(sc - r)
            d = [0, 0, 0, 0, 1, 1, 0, 0]

        if np.linalg.norm(sd - r) < d_min:
            d_min = np.linalg.norm(sd - r)
            d = [0, 0, 0, 0, 1, 1, 0, 1]

        if np.linalg.norm(se - r) < d_min:
            d_min = np.linalg.norm(se - r)
            d = [0, 0, 0, 0, 1, 1, 1, 0]

        if np.linalg.norm(sf - r) < d_min:
            d_min = np.linalg.norm(sf - r)
            d = [0, 0, 0, 0, 1, 1, 1, 1]

        if np.linalg.norm(s10 - r) < d_min:
            d_min = np.linalg.norm(s10 - r)
            d = [0, 0, 0, 1, 0, 0, 0, 0]

        if np.linalg.norm(s11 - r) < d_min:
            d_min = np.linalg.norm(s11 - r)
            d = [0, 0, 0, 1, 0, 0, 0, 1]

        if np.linalg.norm(s12 - r) < d_min:
            d_min = np.linalg.norm(s12 - r)
            d = [0, 0, 0, 1, 0, 0, 1, 0]

        if np.linalg.norm(s13 - r) < d_min:
            d_min = np.linalg.norm(s13 - r)
            d = [0, 0, 0, 1, 0, 0, 1, 1]

        if np.linalg.norm(s14 - r) < d_min:
            d_min = np.linalg.norm(s14 - r)
            d = [0, 0, 0, 1, 0, 1, 0, 0]

        if np.linalg.norm(s15 - r) < d_min:
            d_min = np.linalg.norm(s15 - r)
            d = [0, 0, 0, 1, 0, 1, 0, 1]

        if np.linalg.norm(s16 - r) < d_min:
            d_min = np.linalg.norm(s16 - r)
            d = [0, 0, 0, 1, 0, 1, 1, 0]

        if np.linalg.norm(s17 - r) < d_min:
            d_min = np.linalg.norm(s17 - r)
            d = [0, 0, 0, 1, 0, 1, 1, 1]

        if np.linalg.norm(s18 - r) < d_min:
            d_min = np.linalg.norm(s18 - r)
            d = [0, 0, 0, 1, 1, 0, 0, 0]

        if np.linalg.norm(s19 - r) < d_min:
            d_min = np.linalg.norm(s19 - r)
            d = [0, 0, 0, 1, 1, 0, 0, 1]

        if np.linalg.norm(s1a - r) < d_min:
            d_min = np.linalg.norm(s1a - r)
            d = [0, 0, 0, 1, 1, 0, 1, 0]

        if np.linalg.norm(s1b - r) < d_min:
            d_min = np.linalg.norm(s1b - r)
            d = [0, 0, 0, 1, 1, 0, 1, 1]

        if np.linalg.norm(s1c - r) < d_min:
            d_min = np.linalg.norm(s1c - r)
            d = [0, 0, 0, 1, 1, 1, 0, 0]

        if np.linalg.norm(s1d - r) < d_min:
            d_min = np.linalg.norm(s1d - r)
            d = [0, 0, 0, 1, 1, 1, 0, 1]

        if np.linalg.norm(s1e - r) < d_min:
            d_min = np.linalg.norm(s1e - r)
            d = [0, 0, 0, 1, 1, 1, 1, 0]

        if np.linalg.norm(s1f - r) < d_min:
            d_min = np.linalg.norm(s1f - r)
            d = [0, 0, 0, 1, 1, 1, 1, 1]

        if np.linalg.norm(s20 - r) < d_min:
            d_min = np.linalg.norm(s20 - r)
            d = [0, 0, 1, 0, 0, 0, 0, 0]

        if np.linalg.norm(s21 - r) < d_min:
            d_min = np.linalg.norm(s21 - r)
            d = [0, 0, 1, 0, 0, 0, 0, 1]

        if np.linalg.norm(s22 - r) < d_min:
            d_min = np.linalg.norm(s22 - r)
            d = [0, 0, 1, 0, 0, 0, 1, 0]

        if np.linalg.norm(s23 - r) < d_min:
            d_min = np.linalg.norm(s23 - r)
            d = [0, 0, 1, 0, 0, 0, 1, 1]

        if np.linalg.norm(s24 - r) < d_min:
            d_min = np.linalg.norm(s24 - r)
            d = [0, 0, 1, 0, 0, 1, 0, 0]

        if np.linalg.norm(s25 - r) < d_min:
            d_min = np.linalg.norm(s25 - r)
            d = [0, 0, 1, 0, 0, 1, 0, 1]

        if np.linalg.norm(s26 - r) < d_min:
            d_min = np.linalg.norm(s26 - r)
            d = [0, 0, 1, 0, 0, 1, 1, 0]

        if np.linalg.norm(s27 - r) < d_min:
            d_min = np.linalg.norm(s27 - r)
            d = [0, 0, 1, 0, 0, 1, 1, 1]

        if np.linalg.norm(s28 - r) < d_min:
            d_min = np.linalg.norm(s28 - r)
            d = [0, 0, 1, 0, 1, 0, 0, 0]

        if np.linalg.norm(s29 - r) < d_min:
            d_min = np.linalg.norm(s29 - r)
            d = [0, 0, 1, 0, 1, 0, 0, 1]

        if np.linalg.norm(s2a - r) < d_min:
            d_min = np.linalg.norm(s2a - r)
            d = [0, 0, 1, 0, 1, 0, 1, 0]

        if np.linalg.norm(s2b - r) < d_min:
            d_min = np.linalg.norm(s2b - r)
            d = [0, 0, 1, 0, 1, 0, 1, 1]

        if np.linalg.norm(s2c - r) < d_min:
            d_min = np.linalg.norm(s2c - r)
            d = [0, 0, 1, 0, 1, 1, 0, 0]

        if np.linalg.norm(s2d - r) < d_min:
            d_min = np.linalg.norm(s2d - r)
            d = [0, 0, 1, 0, 1, 1, 0, 1]

        if np.linalg.norm(s2e - r) < d_min:
            d_min = np.linalg.norm(s2e - r)
            d = [0, 0, 1, 0, 1, 1, 1, 0]

        if np.linalg.norm(s2f - r) < d_min:
            d_min = np.linalg.norm(s2f - r)
            d = [0, 0, 1, 0, 1, 1, 1, 1]

        if np.linalg.norm(s30 - r) < d_min:
            d_min = np.linalg.norm(s30 - r)
            d = [0, 0, 1, 1, 0, 0, 0, 0]

        if np.linalg.norm(s31 - r) < d_min:
            d_min = np.linalg.norm(s31 - r)
            d = [0, 0, 1, 1, 0, 0, 0, 1]

        if np.linalg.norm(s32 - r) < d_min:
            d_min = np.linalg.norm(s32 - r)
            d = [0, 0, 1, 1, 0, 0, 1, 0]

        if np.linalg.norm(s33 - r) < d_min:
            d_min = np.linalg.norm(s33 - r)
            d = [0, 0, 1, 1, 0, 0, 1, 1]

        if np.linalg.norm(s34 - r) < d_min:
            d_min = np.linalg.norm(s34 - r)
            d = [0, 0, 1, 1, 0, 1, 0, 0]

        if np.linalg.norm(s35 - r) < d_min:
            d_min = np.linalg.norm(s35 - r)
            d = [0, 0, 1, 1, 0, 1, 0, 1]

        if np.linalg.norm(s36 - r) < d_min:
            d_min = np.linalg.norm(s36 - r)
            d = [0, 0, 1, 1, 0, 1, 1, 0]

        if np.linalg.norm(s37 - r) < d_min:
            d_min = np.linalg.norm(s37 - r)
            d = [0, 0, 1, 1, 0, 1, 1, 1]

        if np.linalg.norm(s38 - r) < d_min:
            d_min = np.linalg.norm(s38 - r)
            d = [0, 0, 1, 1, 1, 0, 0, 0]

        if np.linalg.norm(s39 - r) < d_min:
            d_min = np.linalg.norm(s39 - r)
            d = [0, 0, 1, 1, 1, 0, 0, 1]

        if np.linalg.norm(s3a - r) < d_min:
            d_min = np.linalg.norm(s3a - r)
            d = [0, 0, 1, 1, 1, 0, 1, 0]

        if np.linalg.norm(s3b - r) < d_min:
            d_min = np.linalg.norm(s3b - r)
            d = [0, 0, 1, 1, 1, 0, 1, 1]

        if np.linalg.norm(s3c - r) < d_min:
            d_min = np.linalg.norm(s3c - r)
            d = [0, 0, 1, 1, 1, 1, 0, 0]

        if np.linalg.norm(s3d - r) < d_min:
            d_min = np.linalg.norm(s3d - r)
            d = [0, 0, 1, 1, 1, 1, 0, 1]

        if np.linalg.norm(s3e - r) < d_min:
            d_min = np.linalg.norm(s3e - r)
            d = [0, 0, 1, 1, 1, 1, 1, 0]

        if np.linalg.norm(s3f - r) < d_min:
            d_min = np.linalg.norm(s3f - r)
            d = [0, 0, 1, 1, 1, 1, 1, 1]

        #Conteo de errores:
        if d[0] != sm[0]:
            errores = errores + 1

        if d[1] != sm[1]:
            errores = errores + 1

        if d[2] != sm[2]:
            errores = errores + 1

        if d[3] != sm[3]:
            errores = errores + 1

        if d[4] != sm[4]:
            errores = errores + 1

        if d[5] != sm[5]:
            errores = errores + 1

        if d[6] != sm[6]:
            errores = errores + 1

        if d[7] != sm[7]:
            errores = errores + 1

    #Estimacion de la probabilidad de error:
    p = errores / float(8*N)
    return p


if __name__ == '__main__':
    # Test bench area
    #print("Hola mundo")

    N=10000 #Numero de experimentos de la simulacion Monte Carlo
    snr_in_dB = np.arange(0.0, 25.0, 1.0)

    Pe = np.zeros(len(snr_in_dB))
    for i in range(len(snr_in_dB)):
        Pe[i]=miSimPe_64_QAM_sys(snr_in_dB[i], N)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Graficacion:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.semilogy(snr_in_dB,Pe)
    plt.title('Pe para deteccion 64-QAM con simulacion Monte Carlo')
    plt.xlabel('SNR en dB')

    plt.show()