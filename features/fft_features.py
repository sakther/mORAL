import sys
import time
import os
import glob
import numpy as np

eps = 0.00000001


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)

def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = np.sum(X ** 2)            # total spectral energy

    subWinLength = int(np.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = np.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -np.sum(s*np.log2(s + eps))                                    # compute spectral entropy

    return En


def stSpectralFlux(X, Xprev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:        the abs(fft) of the current frame
        Xpre:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = np.sum(X + eps)
    sumPrevX = np.sum(Xprev + eps)
    F = np.sum((X / sumX - Xprev/sumPrevX) ** 2)

    return F

def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)

def spectral_entropy(data, sampling_freq, bands=None):

    psd = np.abs(np.fft.rfft(data)) ** 2
    psd /= np.sum(psd)  # psd as a pdf (normalised to one)

    if bands is None:
        power_per_band = psd[psd > 0]
    else:
        freqs = np.fft.rfftfreq(data.size, 1 / float(sampling_freq))
        bands = np.asarray(bands)

        freq_limits_low = np.concatenate([[0.0], bands])
        freq_limits_up = np.concatenate([bands, [np.Inf]])

        power_per_band = [np.sum(psd[np.bitwise_and(freqs >= low, freqs < up)])
                          for low, up in zip(freq_limits_low, freq_limits_up)]

        power_per_band = power_per_band[power_per_band > 0]

    return -np.sum(power_per_band * np.log2(power_per_band))

def fouriar_features(data):
    Fs = 16.0  # the sampling freq (in Hz)

    # fourier transforms!
    data_fft = abs(np.fft.rfft(data))

    X = abs(np.fft.fft(data))
    nFFT = int(len(X)/2)+1

    X = X[0:nFFT]                                    # normalize fft
    X = X / len(X)

    C, S = stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
    se = stSpectralEntropy(X)                  # spectral entropy
    se_old = spectral_entropy(X, 16.0)              # spectral flux
    flx = stSpectralFlux(X, X.copy())              # spectral flux
    roff = stSpectralRollOff(X, 0.90, Fs) # spectral rolloff

    return [C, S, se, se_old, flx, roff]

# def fouriar_features_old(data):
#     # fourier transforms!
#     data_fft = abs(np.fft.rfft(data))
#     # freq_max = np.argmax(data_fft)
#     fft_min = data_fft.min()
#     # power = compute_power(data)
#     # Max Fourier
#     fft_max = data_fft.max()
#
#     # Mean Fourier
#     fft_mean= data_fft.mean()
#
#     # Standard deviation Fourier
#     fft_std = data_fft.std()
#
#     fft_kurtosis = kurtosis(data_fft)
#
#     spec_entropy= spectral_entropy(data, 16.0)
#
#     return [fft_min, fft_max, fft_mean, fft_std, fft_kurtosis, spec_entropy]
