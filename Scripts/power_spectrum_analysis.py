# -*- coding: utf-8 -*-
from __future__ import print_function, division

import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy import stats
from scipy.optimize import minimize


def sinc2(x, a, b, c):
    return a * np.sinc((x - b) / c) ** 2 + 2


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)) + 2


def MLERegression(parameters, x, y, noise_pds, which_type):
    eps = 10 ** -20
    if which_type == "Gaussian":
        mu, sigma = parameters[0], parameters[1]
        yhat = gaussian(x, mu, sigma)  # + noise_pds
    elif which_type == "Sinc^2":
        a, b, c = parameters[0], parameters[1], parameters[2]
        yhat = sinc2(x, a, b, c)  # + noise_pds
    negLL = 2 * np.sum((y / (yhat + eps) + np.log(yhat + eps)))
    return negLL ** 2


def fit_sinc2(dft_amplitude, freq, noise_pds, count=500):
    guess = np.array([2, 1, 5])
    result_min = minimize(MLERegression, guess,
                          args=(freq, dft_amplitude, noise_pds, "Sinc^2"), method='Nelder-Mead',
                          options={'adaptive': True})
    for i in range(count - 1):
        guess = np.random.rand(3) * 10
        result_tmp = minimize(MLERegression, guess,
                              args=(freq, dft_amplitude, noise_pds, "Sinc^2"), method='Nelder-Mead',
                              options={'adaptive': True})
        if MLERegression(result_tmp.x, freq, dft_amplitude, noise_pds, 'Sinc^2') < MLERegression(result_min.x, freq,
                                                                                                 dft_amplitude,
                                                                                                 noise_pds,
                                                                                                 'Sinc^2'):
            result_min = result_tmp
    return result_min


def fit_gaussian(dft_amplitude, freq, noise_pds, count=2000):
    guess = np.array([0.38, 0.4])
    result_min = minimize(MLERegression, guess,
                          args=(freq, dft_amplitude, noise_pds, "Gaussian"), method='Nelder-Mead',
                          options={'adaptive': True})
    for i in range(count - 1):
        guess = np.random.rand(2) * 10
        result_tmp = minimize(MLERegression, guess,
                              args=(freq, dft_amplitude, noise_pds, "Gaussian"), method='Nelder-Mead',
                              options={'adaptive': True})
        if MLERegression(result_tmp.x, freq, dft_amplitude, noise_pds, 'Gaussian') < MLERegression(result_min.x, freq,
                                                                                                   dft_amplitude,
                                                                                                   noise_pds,
                                                                                                   'Gaussian'):
            result_min = result_tmp
    return result_min


def plot_fitting(freq, dft_amplitude, noise_pds):
    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    results_gaussian = fit_gaussian(dft_amplitude[1:], freq[1:], noise_pds[1:])
    axs[0].plot(freq[1:], dft_amplitude[1:], freq[1:], gaussian(freq[1:], results_gaussian.x[0], results_gaussian.x[1]),
                "g--")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Power")
    axs[0].set_title("Gaussian fitting")

    results_sinc2 = fit_sinc2(dft_amplitude[1:], freq[1:], noise_pds[1:])
    axs[1].plot(freq[1:], dft_amplitude[1:], freq[1:],
                sinc2(freq[1:], results_sinc2.x[0], results_sinc2.x[1], results_sinc2.x[2]), "r--")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Power")
    axs[1].set_title("Sinc^2 fitting")

    plt.show()


def plot_signal(arr_time, signal, freq, dft_amplitude, noise_pds):
    N = arr_time.size
    plt.subplot(1, 2, 1)
    plt.plot(arr_time, signal)

    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # print (freq)
    plt.subplot(1, 2, 2)
    plt.plot(freq[1:], dft_amplitude[1:], "k-", marker="o", mfc='none')

    plt.plot(freq, noise_pds, linewidth=0.75)

    P_3sig = 0.00135
    P_5sig = 2.87e-7

    sig_lev = stats.expon.ppf(1 - P_3sig, scale=noise_pds[0])  # 3 sigma
    plt.plot(freq, sig_lev * np.ones_like(freq), 'g--', linewidth=1)
    # print("3 sig level: ", sig_lev)

    sig_lev = stats.expon.ppf(1 - P_5sig, scale=noise_pds[0])  # 5 sigma
    plt.plot(freq, sig_lev * np.ones_like(freq), 'r--', linewidth=1)
    # print("5 sig level: ", sig_lev)

    plt.xlabel("Freq (Hz)")
    plt.ylabel("Power")

    plt.show()


def signal_with_noise():
    # N = 1024
    N = 256

    sample_rate = 2.0  # измерений сигнала в секунду (Гц)
    sample_spacing = 1.0 / sample_rate  # расстояние между измерениями

    harmonic_amp = 3.0
    #    signal_f = [0.1, 0.2, 0.33, 0.55, 1, 0.75]
    signal_f = [0.38]
    arr_time = np.array([t for t in range(N)])
    signal = 20.0

    for i in signal_f:
        signal += harmonic_amp * np.sin(2.0 * np.pi * i * arr_time / sample_rate)

    noise = np.array([random.gauss(0.0, signal[i] ** 0.5) for i in range(N)])  # Генерируем шум
    signal = signal + noise

    return signal, arr_time, sample_rate


def main():
    signal, arr_time, sample_rate = signal_with_noise()

    N_gamma = np.sum(signal)
    signal_var = signal - np.mean(signal)
    dft_amplitude = 2 * np.abs(rfft(signal_var)) ** 2 / N_gamma

    N = arr_time.size
    freq = rfftfreq(N, d=1 / sample_rate)
    noise_pds = 2.0 * np.ones_like(freq)

    # plot_signal(arr_time, signal, freq, dft_amplitude, noise_pds)

    plot_fitting(freq, dft_amplitude, noise_pds)


if __name__ == "__main__":
    main()
