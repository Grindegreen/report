# -*- coding: utf-8 -*-

import numpy as np


def fix_gaps(arr_ti, arr_tf, arr_cnts):
    step = 2.944
    arr_ti_fx = [arr_ti[0], ]
    arr_tf_fx = [arr_tf[0], ]
    arr_cnt_fx = [arr_cnts[0], ]

    eps = step / 2

    n_extra = 0
    for i in xrange(1, arr_ti.size):
        if np.abs(arr_ti[i] - arr_tf[i - 1]) < eps:
            arr_ti_fx.append(arr_ti[i])
            arr_tf_fx.append(arr_tf[i])
            arr_cnt_fx.append(arr_cnts[i])

        else:
            # print i, repr(arr_ti[i]), repr(arr_tf[i-1]),  repr(arr_ti[i] - arr_tf[i-1])

            k = 0
            while (arr_ti[i] - arr_tf_fx[i - 1 + k + n_extra]) >= eps:
                # print repr(arr_tf_fx[i-1+k+n_extra]), repr(arr_ti[i]), repr(arr_tf_fx[i-1+k+n_extra] - arr_ti[i])
                arr_ti_fx.append(arr_tf[i - 1] + k * step)
                arr_tf_fx.append(arr_tf[i - 1] + (k + 1) * step)
                arr_cnt_fx.append(np.nan)
                k += 1

            n_extra += k
            arr_tf_fx[-1] = arr_ti[i]
            arr_ti_fx.append(arr_ti[i])
            arr_tf_fx.append(arr_tf[i])
            arr_cnt_fx.append(arr_cnts[i])
            # print ''

    return np.array(arr_ti_fx), np.array(arr_tf_fx), np.array(arr_cnt_fx)


def fix_glitches(arr_ti, arr_tf, arr_cnts):
    """
    Удаление гличей. Предполагается, что в первых n_sum нет NaN
    """

    n_sum = 500

    idx_nan = np.argwhere(np.isnan(arr_cnts[0:n_sum])).flatten()
    n_nans = idx_nan.size

    if n_nans > 0:
        print "NaNs in the beginning should be fixed manually! Exiting..."
        exit()

    lst_glitches = []
    for i in xrange(n_sum + 1, arr_cnts.size):

        mean = np.sum(arr_cnts[i - n_sum - 1:i - 1]) / n_sum

        if np.isnan(arr_cnts[i]):
            arr_cnts[i] = mean

        if np.abs(arr_cnts[i] - mean) > 4.5 * mean ** 0.5:
            lst_glitches.append((i, arr_ti[i], arr_tf[i], arr_cnts[i], mean))
            arr_cnts[i] = mean

    with open("glitches.log", 'w') as f:
        for g in lst_glitches:
            f.write(" ".join(map(str, g)) + "\n")

    return arr_cnts


def clean_data(file_name, channel):
    """
    читаем thc для одного детектора : ti tf G1, G2, G3, Z и 
    выводим thc для одного канала с заполненными пропусками временных бинов
    """

    dict_channels = {'G1': 2, 'G2': 3, 'G3': 4, 'Z': 5}
    data = np.genfromtxt("{:s}.thc".format(file_name), skip_header=0)

    # data = data[:10000,:]

    T0_mjd = data[0, 0]
    arr_ti = (data[:, 0] - T0_mjd) * 86400
    arr_tf = (data[:, 1] - T0_mjd) * 86400
    arr_counts = data[:, dict_channels[channel]]

    # bins with NaNs may have wrong timing, i.e. arr_tf
    arr_bool = np.logical_not(np.isnan(arr_counts))
    arr_ti = arr_ti[arr_bool]
    arr_tf = arr_tf[arr_bool]
    arr_counts = arr_counts[arr_bool]

    eps = 1e-4
    if 0:
        # print np.array2string(arr_tf - arr_ti ,threshold=10000)
        arr_bool = np.abs(arr_tf - arr_ti - 2.944) > eps
        print np.argwhere(arr_bool)
        tis = arr_ti[np.nonzero(arr_bool)] / 86400 + T0_mjd
        arr_res = np.around(arr_tf[np.nonzero(arr_bool)] - arr_ti[np.nonzero(arr_bool)], 3)
        # print arr_res
        nst_bins = np.stack((tis, arr_res, arr_counts[np.nonzero(arr_bool)]), axis=-1)
        np.savetxt('kw58175_58350mjd_nonst_bins.txt', nst_bins, delimiter=' ', fmt=['%16.10f', '%8.3f', '%7.2f'])
        # exit()

    arr_ti_fx, arr_tf_fx, arr_cnt_fx = fix_gaps(arr_ti, arr_tf, arr_counts)
    gaps = arr_ti_fx[1:] - arr_tf_fx[:-1]
    gaps = np.append(gaps, 0)
    print "Nonstandard bin indexes: \n", np.argwhere(np.abs(gaps) > eps)
    print arr_ti_fx[np.abs(gaps) > eps]

    # data = np.stack((arr_ti_fx/86400+T0_mjd, arr_tf_fx/86400+T0_mjd, arr_cnt_fx), axis=-1)
    # np.savetxt("{:s}_{:s}_fixed_bins.thc".format(file_name, channel), data, delimiter=' ',fmt=['%16.10f','%16.10f', '%7.2f'])

    # Fix glitches and NaNs
    arr_cnt_fx = fix_glitches(arr_ti_fx / 86400 + T0_mjd, arr_tf_fx / 86400 + T0_mjd, arr_cnt_fx)

    data = np.stack((arr_ti_fx / 86400 + T0_mjd, arr_tf_fx / 86400 + T0_mjd, arr_cnt_fx), axis=-1)
    np.savetxt("{:s}_{:s}_fixed.thc".format(file_name, channel), data, delimiter=' ',
               fmt=['%16.10f', '%16.10f', '%7.2f'])


def main():
    det = '2'
    channel = ['G1', 'G2', 'G3', 'Z']
    for i in xrange(0, 4):
        file_name = 'kwS{:s}_58160_58350mjd'.format(det)

        clean_data(file_name, channel[i])


main()
