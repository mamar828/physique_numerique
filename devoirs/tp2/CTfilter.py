#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np

## filtrer le sinogramme
## ligne par ligne
def filterSinogram(sinogram):
    for i in range(sinogram.shape[0]):
        sinogram[i] = filterLine(sinogram[i])

## filter une ligne (projection) via FFT
def filterLine(projection):

    # votre code ici
    # un filtre rampe est suffisant 
    freq = np.fft.rfft(projection)
    freq_filtrees = freq*np.abs(np.fft.rfftfreq(len(projection)))
    ligne_filtree = np.fft.irfft(freq_filtrees)

    return ligne_filtree
