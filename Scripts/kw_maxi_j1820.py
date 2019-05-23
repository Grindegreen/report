# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import t

#plt.rc('font',family='serif', serif='Arial')
plt.rc('axes',labelsize=14)
plt.rc('axes',titlesize=14)
plt.rc('xtick', direction='in', top=True, labelsize=14)
plt.rc('ytick', direction='in', right=True, labelsize=14)
plt.rc('figure')#, figsize = (8.27, 11.69))

def mean(x, confidence=0.68):
    n = x.size
    m, se = np.mean(x), np.std(x)
    m_err = se * t.ppf((1+confidence)/2., n-1)
    return m, m_err

def poly3(p, x, x0):
    x = x-x0
    return p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x # Target function

def rise(p, x, x0):
    x = x-x0
    x = x-p[0]
    return p[1] + (np.sign(x)+1.0)/2.0 * (p[2]*x + p[3]*x*x)

def func(p, x, x0):
    return rise(p, x, x0)

def errfunc(p, x, x0, y):
    #return (poly3(p, x, x0) - y)/y**0.5
    return (func(p, x, x0) - y)/y**0.5

def fit_bg(arr_ti, arr_cnts):

    nskip = arr_ti.size/1000
    #nskip = 2

    x = arr_ti[::nskip]
    y = arr_cnts[::nskip]

    #p0 = [3300, 0.1, 0.1, 0.01] # Initial guess for the parameters
    p0 = [5, 1200, 1, 0.1] # G2
    #p0 = [5, 3300, 1.0, 0.1] # G1

    out = optimize.leastsq(errfunc, p0[:], args=(x,0,y))

    pfinal = out[0]

    print "chi2/dof={:8.3f}/{:d}".format(np.sum(errfunc(pfinal,x,0, y)**2), x.size)
    print "par: ", out

    return pfinal

def running_mean(ti, tf, x, N):

    dt = tf[N-1::N] - ti[:-N+1:N]
    #print tf[N-1::N], ti[:-N+1:N], dt

    cumsum = np.cumsum(np.insert(x, 0, 0))
    #print cumsum[N::N] - cumsum[:-N:N]
    rate = (cumsum[N::N] - cumsum[:-N:N]) / dt
    rate_err = (cumsum[N::N] - cumsum[:-N:N])**0.5 / dt
    ti_out = ti[:-N+1:N]
    tf_out = tf[N-1::N]
    return ti_out, tf_out, rate, rate_err

def read_kw_data(file_name):
    """
    читаем thc для одного детектора и одного канала: ti tf Gi
    """

    #n_sum = 3397 # 3397 * 2.944 = 10 ks
    n_sum = 14674 # 12 h
    data = np.genfromtxt(file_name, skip_header=0)

    #arr_bool = data[:,0] > 58175
    #data = data[arr_bool,:]

    T0_mjd = data[0,0]
    arr_ti = (data[:,0] - T0_mjd)*86400
    arr_tf = (data[:,1] - T0_mjd)*86400
    arr_counts = data[:,2]

    ti_out, tf_out, rate, rate_err = running_mean(arr_ti, arr_tf, arr_counts, n_sum)

    return ti_out, tf_out, rate, rate_err, T0_mjd

def plot_kw_wm(arr_ti, arr_tf, arr_rate, arr_rate_err, det, channel):

    t_cnt = (arr_ti + arr_tf)/2.0
    dt = (arr_tf - arr_ti)/2.0

    plt.errorbar(t_cnt, arr_rate, xerr=dt, yerr=arr_rate_err, fmt='ro', capsize=0, markersize=3)

    plt.xlabel("Time(MJD)")
    plt.ylabel("S{:s} {:s} rate (counts/s)".format(det,channel))
    plt.title('MAXI J1820')
    plt.grid(True)
    plt.savefig('MAXIJ1820_kw_S{:s}{:s}.png'.format(det,channel))
    #plt.show()

def plot_kw_wm_S12(arr_ti, arr_tf, arr_rate_s1, arr_rate_err_s1, arr_rate_s2, arr_rate_err_s2, channel):

    dic_scale = {'G1': 1.1, 'G2': 1.2,'G3': 0.8, 'Z': 1.1}

    t_cnt = (arr_ti + arr_tf)/2.0
    dt = (arr_tf - arr_ti)/2.0

    scale = dic_scale[channel]
    arr_rate_s1 *= scale
    arr_rate_err_s1 *= scale

    plt.figure(1)
    plt.subplot(311)
    plt.errorbar(t_cnt, arr_rate_s1, xerr=dt, yerr=arr_rate_err_s1, fmt='ro', capsize=0, markersize=3, label="S1 x {:5.1f}".format(scale))
    plt.errorbar(t_cnt, arr_rate_s2, xerr=dt, yerr=arr_rate_err_s2, fmt='bo', capsize=0, markersize=3, label="S2")

    #plt.xlabel("Time(MJD)")
    plt.ylabel("{:s} rate (counts/s)".format(channel))
    plt.title('Konus-Wind waiting mode')
    plt.grid(True)
    plt.legend()

    plt.subplot(312)
    plt.scatter(t_cnt, arr_rate_s1/arr_rate_s2/scale, s=3**0.5)

    arr_bool = t_cnt < 58180 # mjd
    m = np.mean(arr_rate_s1[arr_bool]/arr_rate_s2[arr_bool]/scale)
    print "S1/S2 mean= ",m
    plt.axhline(y=m, ls='--')

    #plt.xlabel("Time(MJD)")
    plt.ylabel("S1/S2")
    #plt.ylim(0.98,1)
    plt.grid(True)

    plt.subplot(313)

    r_bg_sub = arr_rate_s2 - arr_rate_s1/scale/m
    r_bg_sub_err = (arr_rate_err_s2**2 + m**2*arr_rate_err_s1**2)**0.5
    plt.errorbar(t_cnt, r_bg_sub, xerr=dt, yerr=r_bg_sub_err, fmt='bo', capsize=0, markersize=3, label="S2")
    plt.axhline(y=0, ls='--')

    plt.ylabel("{:s} bg sub rate counts/s".format(channel))
    plt.xlabel("Time(MJD)")
    plt.grid(True)
    plt.savefig('kw_wm_{:s}.png'.format(channel))

def plot_kw_other(arr_ti, arr_tf, arr_rate, arr_rate_err, channel, o_tc, o_dt, o_rate, o_rate_err, other_label):

    dic_y_range = {'G1':(-10, 110), 'G2':(-5, 25)}

    y_min = dic_y_range[channel][0]
    y_max = dic_y_range[channel][1]
#    x_step = 40

    fig, ax1 = plt.subplots()#figsize = (8.27, 8.27))

    t_cnt = (arr_ti + arr_tf)/2.0
    dt = (arr_tf - arr_ti)/2.0

    ax1.errorbar(t_cnt, arr_rate, xerr=dt, yerr=arr_rate_err, fmt='bo', capsize=0, markersize=3)
    ax1.set_xlabel('Time (MJD)')
    ax1.axhline(y=0.0, ls='--')
    ax1.set_ylim(y_min, y_max)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("{:s} rate (counts/s)".format(channel), color='b')
    ax1.tick_params('y', colors='b')

#    ax1.xaxis.set_ticks(np.arange(58160, 58210+x_step, x_step))

    scale = np.max(arr_rate) / np.max(o_rate)
    o_rate *= scale
    o_rate_err *= scale

    ax2 = ax1.twinx()
    ax2.errorbar(o_tc, o_rate, xerr=o_dt, yerr=o_rate_err, fmt='ro', capsize=0, markersize=3)
    ax2.set_ylabel(other_label, color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(y_min, y_max)

    plt.title('MAXI J1820')
    plt.grid(True)
    plt.savefig('MAXIJ1820_kw{:s}_int.png'.format(channel))
    plt.show()

def load_bat(file_name):

    data = np.genfromtxt(file_name, skip_header=0)
    return data[:,0], data[:,1], data[:,2], data[:,3]

def load_isgri(file_name):

    data = np.genfromtxt(file_name, skip_header=0)
    return data[:,0], data[:,1], data[:,2], data[:,3]

def test_rate():

    ti  = np.array([1, 2,3,4,5,6,7,8])
    tf  = np.array([2, 3,4,5,6,7,8,9])
    cnt = np.array([1.0,1,2,1,3,1,1,1])
    ti_out, tf_out, rate, rate_err = running_mean(ti,tf,cnt, 3)
    print ti_out, tf_out, rate, rate_err

def plot_two_det():

    channel = 'G1'
    file_name_s1 = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format('1', channel)
    file_name_s2 = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format('2', channel)

    arr_ti, arr_tf, arr_rate_s1, arr_rate_err_s1, T0_mjd = read_kw_data(file_name_s1)
    arr_ti, arr_tf, arr_rate_s2, arr_rate_err_s2, T0_mjd = read_kw_data(file_name_s2)

    arr_ti = arr_ti/86400.0+T0_mjd
    arr_tf = arr_tf/86400.0+T0_mjd

    plot_kw_wm_S12(arr_ti, arr_tf, arr_rate_s1, arr_rate_err_s1, arr_rate_s2, arr_rate_err_s2, channel)


def fix_KW_S2G2():

    channel = 'G2'
    file_name_s1 = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format('1', channel)
    file_name_s2 = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format('2', channel)

    arr_ti, arr_tf, arr_rate_s1, arr_rate_err_s1, T0_mjd = read_kw_data(file_name_s1)
    arr_ti, arr_tf, arr_rate_s2, arr_rate_err_s2, T0_mjd = read_kw_data(file_name_s2)

    arr_ti = arr_ti/86400.0+T0_mjd
    arr_tf = arr_tf/86400.0+T0_mjd

    arr_bool = arr_tf < 58180 # mjd
    m = np.mean(arr_rate_s1[arr_bool]/arr_rate_s2[arr_bool])

    r_bg_sub = arr_rate_s2 - arr_rate_s1/m
    r_bg_sub_err = (arr_rate_err_s2**2 + m**2*arr_rate_err_s1**2)**0.5

    return arr_ti, arr_tf, r_bg_sub, r_bg_sub_err

def main():

    det = '2'
    channel = 'G2'
    file_name = 'kwS{:s}_58160_58350mjd_{:s}_fixed.thc'.format(det, channel)

    do_fix = True

    if do_fix:

        arr_ti, arr_tf, arr_rate, arr_rate_err = fix_KW_S2G2()

    else:
        arr_ti, arr_tf, arr_rate, arr_rate_err, T0_mjd = read_kw_data(file_name)

        arr_ti = arr_ti/86400.0+T0_mjd
        arr_tf = arr_tf/86400.0+T0_mjd

        TiBg = 58175 # MJD
        TfBg = 58350 # MJD
        arr_bool_bg = np.logical_and(arr_ti>TiBg, arr_tf<TfBg)
        bg, err_bg = mean(arr_rate[arr_bool_bg])

        arr_rate -= bg
        arr_rate_err = np.sqrt(arr_rate_err**2 + err_bg**2)

    #plot_data(arr_ti, arr_tf, arr_rate, arr_rate_err, det, channel)

    #file_name = 'MAXIJ1820p070_orbit.txt'
    file_name = 'MAXIJ1820p070_daily.txt'
    bat_ti, bat_dt, bat_rate, bat_rate_err = load_bat(file_name)
    #plot_kw_other(arr_ti, arr_tf, arr_rate, arr_rate_err, channel, bat_ti, bat_dt, bat_rate, bat_rate_err, 'normed Swift (BAT) 15-50 keV')

    file_name = "MAXIJ1820p070_isgri_{:s}_rb.txt".format(channel)
    dic_int = {'G1':'25-60 keV', 'G2':'60-200 keV'}
    int_tc, int_dt, int_rate, int_rate_err = load_isgri(file_name)
    plot_kw_other(arr_ti, arr_tf, arr_rate, arr_rate_err, channel, int_tc, int_dt, int_rate, int_rate_err, 'normed INTEGRAL (ISGRI) {:}'.format(dic_int[channel]))


main()
#plot_two_det()
#test_rate()
