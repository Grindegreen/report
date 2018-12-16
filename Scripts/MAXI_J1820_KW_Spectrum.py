# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy import stats
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

import sys
import warnings

sys.setrecursionlimit(10000)
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def sinc2(x, a, b, c):
    return a * np.sinc((x - b) / c) ** 2 + 2


def gaussian(x, A, mu, sigma):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)) + 2


def noise_plus_gaussian(x, A, mu, sigma, z):
    eps = 10 ** -30
    return gaussian(x, A, mu, sigma) + z / (x + eps)


def MLERegression(parameters, x, y, noise_pds, which_type):
    """
    Функция минимизации - МНП
    parameters - массив предполонаемых параметров фунции
    which_type - какую функцию мы фитируем
    eps нужен для того, чтобы функция не считала логарифм 0
    """
    eps = 10 ** -10
    if which_type == "Gaussian":
        A, mu, sigma = parameters[0], parameters[1], parameters[2]
        yhat = gaussian(x, A, mu, sigma)  # + noise_pds
    elif which_type == "Sinc^2":
        a, b, c = parameters[0], parameters[1], parameters[2]
        yhat = sinc2(x, a, b, c)  # + noise_pds
    elif which_type == "Noise_plus_Gaussian":
        A, mu, sigma, z = parameters[0], parameters[1], parameters[2], parameters[3]
        yhat = noise_plus_gaussian(x, A, mu, sigma, z)  # + noise_pds
    negLL = 2 * np.sum((y / (yhat + eps) + np.log(yhat + eps)))
    return negLL


def fit_sinc2(dft_amplitude, freq, noise_pds, count=500):
    """
    Функция, ответсвенная за аппрокимацию спектра sinc^2
    count - количество попыток фитинга внутри самой функции
    guess - массив с изначальными параметрами функции
    result_min - результат аппроксимации; он же и возвращается в качестсве результата выполнения
    """
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


def fit_gaussian(dft_amplitude, freq, noise_pds, count=1):
    """
    Функция, ответсвенная за аппрокимацию спектра QPO гауссианой
    count - количество попыток фитинга внутри самой функции
    guess - массив с изначальными параметрами функции
    result_min - результат аппроксимации; он же и возвращается в качестсве результата выполнения
    """
    guess = np.array([1, 0.1, 0.1, 0.1])
    result_min = minimize(MLERegression, guess,
                          args=(freq, dft_amplitude, noise_pds, "Noise_plus_Gaussian"), method='Nelder-Mead',
                          options={'adaptive': True})
    for i in range(count - 1):
        # guess = np.array([50, 0.038, 0, 0])
        # guess[2], guess[3] = np.random.rand(2)
        # guess[2] *= 10
        guess = (np.random.rand(4) + 1) * 100
        result_tmp = minimize(MLERegression, guess,
                              args=(freq, dft_amplitude, noise_pds, "Noise_plus_Gaussian"), method='Nelder-Mead',
                              options={'adaptive': True})
        if MLERegression(result_tmp.x, freq, dft_amplitude, noise_pds, 'Noise_plus_Gaussian') < MLERegression(
                result_min.x, freq, dft_amplitude, noise_pds, 'Noise_plus_Gaussian'):
            result_min = result_tmp
    return result_min


def plot_fitting(freq, dft_amplitude, noise_pds, day):
    """
    Функция построения аппроксимированной функции поверх спектра мощности
    day - день, в который построен спектр; нужен, для сохранения графиков
    """
    fig, axs = plt.subplots(1, 1, constrained_layout=True)

    results_gaussian = fit_gaussian(dft_amplitude[1:], freq[1:], noise_pds[1:])
    axs.plot(freq, dft_amplitude, freq,
             noise_plus_gaussian(freq, results_gaussian.x[0], results_gaussian.x[1], results_gaussian.x[2],
                                 results_gaussian.x[3]), "g--")
    plt.xscale('log')
    axs.set_xlabel("Frequency (Hz)")
    axs.set_ylabel("Power")
    axs.set_title("Gaussian fitting")

    plt.savefig('figures\\MAXI_J1820+070_KW_gaussian_approximation_day_{:d}.png'.format(day))


'''    results_sinc2 = fit_sinc2(dft_amplitude[1:], freq[1:], noise_pds[1:])
    axs[1].plot(freq[1:], dft_amplitude[1:], freq[1:],
                sinc2(freq[1:], results_sinc2.x[0], results_sinc2.x[1], results_sinc2.x[2]), "r--")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Power")
    axs[1].set_title("Sinc^2 fitting")
'''


def plot_signal(arr_time, signal, freq, dft_amplitude, noise_pds, grid):
    # N = arr_time.size
    # plt.subplot(grid[0, 0])
    # plt.plot(arr_time, signal)
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")

    #    print(N, N // 2, freq.size)
    # print (freq)
    # plt.subplot(grid[0, 1:])
    plt.subplot(grid[0, 0:])

    plt.plot(freq[1:], dft_amplitude[1:], "k-", marker="o", mfc='none')

    plt.plot(freq, noise_pds, linewidth=0.75)

    P_3sig = 0.00135
    P_5sig = 2.87e-7

    sig_lev = stats.expon.ppf(1 - P_3sig, scale=noise_pds[0])  # 3 sigma
    plt.plot(freq, sig_lev * np.ones_like(freq), 'g--', linewidth=1)
    #    print("3 sig level: ", sig_lev)

    plt.xscale('log')

    sig_lev = stats.expon.ppf(1 - P_5sig, scale=noise_pds[0])  # 5 sigma
    plt.plot(freq, sig_lev * np.ones_like(freq), 'r--', linewidth=1)
    #    print("5 sig level: ", sig_lev)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.xlabel("Freq (Hz)")
    plt.ylabel("Power")

    plt.show()


def plot_spec_distr(spectrum, noise_pds, grid):
    # print (spectrum.size)
    # print (spectrum)
    N = spectrum.size

    plt.subplot(grid[1, 0:])
    n, bins, patches = plt.hist(spectrum, bins=int(N ** 0.5), normed=True)
    # print (n, bins, patches)

    bin_centers = bins + (bins[1] - bins[0]) / 2.0
    y = np.exp(
        -bin_centers / 2) / 2  # Написать верное выражение для функции плотности распределения спектра мощности белого шума

    # print(np.sum(n), np.sum(y))
    plt.plot(bin_centers, y, 'r--', linewidth=2)

    x = stats.expon.rvs(scale=noise_pds[0], size=N)
    print("KS-test results: ", stats.ks_2samp(spectrum, x))

    plt.xlabel('Power')
    plt.ylabel('Number of freq.')
    # plt.title('Power PDF')
    plt.xlim((0, 30))
    plt.show()


def print_pds(freq, dft_amplitude):
    fout = open("data.txt", "w")

    for i in range(freq.size):
        print(freq[i], dft_amplitude[i], file=fout)
    fout.close()


def print_gaussian(day, result, fout):
    """
    Функция, в которой производится вывод параметров функции в файл
    """
    # fout.write("A (coefficent by Gaussian)  mu (mean)  sigma   z( z / f )")
    print(day, result.x[0], result.x[1], result.x[2], result.x[3],
          noise_plus_gaussian(result.x[1], result.x[0], result.x[1], result.x[2], result.x[3]), sep="    ", file=fout)


def running_mean(ti, tf, x, N):
    dt = tf[N - 1::N] - ti[:-N + 1:N]
    # print tf[N-1::N], ti[:-N+1:N], dt

    cumsum = np.cumsum(np.insert(x, 0, 0))
    # print cumsum[N::N] - cumsum[:-N:N]
    rate = (cumsum[N::N] - cumsum[:-N:N]) / dt
    rate_err = (cumsum[N::N] - cumsum[:-N:N]) ** 0.5 / dt
    ti_out = ti[:-N + 1:N]
    tf_out = tf[N - 1::N]
    return ti_out, tf_out, rate, rate_err


def read_kw_data(file_name):
    """
    читаем thc для одного детектора и одного канала: ti tf Gi
    """

    # n_sum = 3397 # 3397 * 2.944 = 10 ks
    n_sum = 14674  # 12 h
    data = np.genfromtxt(file_name, skip_header=0)

    # arr_bool = data[:,0] > 58175
    # data = data[arr_bool,:]

    T0_mjd = data[0, 0]
    arr_ti = (data[:, 0] - T0_mjd) * 86400
    arr_tf = (data[:, 1] - T0_mjd) * 86400
    arr_counts = data[:, 2]

    # ti_out, tf_out, rate, rate_err = running_mean(arr_ti, arr_tf, arr_counts, n_sum)
    return arr_ti, arr_tf, arr_counts, None, T0_mjd
    # return ti_out, tf_out, rate, rate_err, T0_mjd


def gaussian_parameters(file_name):
    """
    Основная функция, в которой спектр разбивается на дни.
    Для каждого из них находится пик QPO и его выводится в файл, а затем в график
    """
    ti_out, tf_out, counts_out, rate_err, T0_mjd = read_kw_data(file_name)

    LENGTH = len(ti_out)

    # sample_rate = 2
    sample_rate = 1 / (tf_out[1] - ti_out[1])
    j0 = 0
    # while ti_out[j0] / 86400 < 40:
    #     j0 += 1
    fout = open("Gaussian parametres.txt", "w")

    """
    Формула для аппроксимации: A * gaussian(mu, sigma) + z / x
    """
    fout.write("day   A    mu   sigma   z   Power\n")

    for i in range(130):
        """
        Последовательное разбиение спектра на дни
        """
        signal = []
        arr_time = []
        while i <= ti_out[j0] / 86400 <= (i + 1):
            signal += [counts_out[j0]]
            arr_time += [ti_out[j0]]
            j0 += 1

        signal = np.array(signal)
        arr_time = np.array(arr_time)
        N_gamma = np.sum(signal)
        signal_var = signal - np.mean(signal)
        dft_amplitude = 2 * np.abs(rfft(signal_var)) ** 2 / N_gamma

        N = arr_time.size
        freq = rfftfreq(N, d=1 / sample_rate)
        noise_pds = 2.0 * np.ones_like(freq)

        # grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)
        # print(dft_amplitude.size, freq.size)
        # exit()
        # plot_signal(arr_time[1:], signal[1:], freq[1:], dft_amplitude[1:], noise_pds[1:], grid)
        # exit()

        # plot_spec_distr(dft_amplitude, noise_pds, grid)
        # print_pds(freq, dft_amplitude)

        plot_fitting(freq[1:], dft_amplitude[1:], noise_pds[1:], i + 1)
        result_gaussian = fit_gaussian(dft_amplitude, freq, noise_pds)
        print_gaussian(i + 1, result_gaussian, fout)

        # print('Parametres:', result_gaussian.x[0], result_gaussian.x[1], result_gaussian.x[2], result_gaussian.x[3],
        #      'Power:', noise_plus_gaussian(result_gaussian.x[1], result_gaussian.x[0], result_gaussian.x[1],
        #                                     result_gaussian.x[2], result_gaussian.x[3]))
    fout.close()


def main():
    det = '2'
    channel = 'G1'

    file_name = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format(det, channel)
    gaussian_parameters(file_name)


if __name__ == "__main__":
    main()
