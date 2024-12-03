import numpy as np
from lmfit import Model


def main():
    print('FakeDataMaker is working!')


def print_info(iqdata_object):
    """
    Print information about the iqdata object.
    """
    print(f'centre frequency: {iqdata_object.center/1e6} MHz')
    print(f"measurement time: {iqdata_object.nsamples_total/iqdata_object.fs} sec")
    print(f'sampling rate: {iqdata_object.fs/1e6} MHz')
    print(f'sample size: {iqdata_object.nsamples_total}')


def import_data(filename, corrected=False, nametag="_corrected"):
    """
    Import isotope's information from csv file.
    """
    data = np.transpose(np.loadtxt(
        f'data/{filename}', delimiter=',', encoding='utf-8-sig'))

    # assign data (nanoseconds to seconds)
    mom_dat = data[-1]
    if corrected:
        freq_dat = 1e9/(data[1])
        isotope_name = filename[:2] + \
            filename[2].capitalize() + filename[3] + nametag
    else:
        freq_dat = 1e9/(data[0])
        isotope_name = filename[:2] + filename[2].capitalize() + filename[3]

    return freq_dat, mom_dat, isotope_name


def gaussian(x, amp, cen, sigma, yoffset=0):
    return (amp * np.exp(-(x-cen)**2 / (2*sigma**2))) + yoffset


def gaussian2(x, amp, cen, sigma):
    return (amp * np.exp(-(x-cen)**2 / (2*sigma**2)))


def get_fit_center(data, x, amp, cen, sigma, yoffset=0, result=False):
    gmodel = Model(gaussian)
    gresult = gmodel.fit(
        data,
        x=x,
        amp=amp,
        cen=cen,
        sigma=sigma,
        yoffset=yoffset
    )
    if result:
        return gresult.params['cen'].value, gresult
    else:
        return gresult.params['cen'].value


def get_fit_center2(data, x, amp, cen, sigma, result=False):
    gmodel = Model(gaussian)
    gresult = gmodel.fit(
        data,
        x=x,
        amp=amp,
        cen=cen,
        sigma=sigma,
    )
    if result:
        return gresult.params['cen'].value, gresult
    else:
        return gresult.params['cen'].value


def get_center_val(x, signal_center):
    center_index = np.argmin(np.abs(x - signal_center))
    return x[center_index]


def create_fake_data(x, freq_data, single_signal, signal_center):
    center_index = np.argmin(np.abs(x - signal_center))
    total_power = np.zeros(len(x))
    for value in freq_data:
        closest_value_index = np.argmin(np.abs(x - value))
        llim = closest_value_index - center_index
        ulim = closest_value_index + len(single_signal) - center_index

        total_power[llim:ulim] += single_signal
    return total_power
