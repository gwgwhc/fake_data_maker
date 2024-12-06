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
    
    Args:
        filename (str): name of the file to import.
        corrected (bool): tag to indicate if the data is corrected or not.
        nametag (str): will be appended to the filename if corrected is True.
        
    Returns:
        freq_dat, mom_dat, isotope_name
    """
    data = np.transpose(np.loadtxt(
        f'data/{filename}', delimiter=',', encoding='utf-8-sig'))

    # assign data (nanoseconds to seconds)
    mom_dat = data[-1]
    if corrected:
        freq_dat = 1e9/(data[1])
        isotope_name = get_isotope_name(filename) + nametag
    else:
        freq_dat = 1e9/(data[0])
        isotope_name = get_isotope_name(filename)

    return freq_dat, mom_dat, isotope_name


def get_isotope_name(filename):
    return filename[:2] + filename[2].capitalize() + filename[3]


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
    """
    version without yoffset
    """
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


def get_index(array, value):
    """
    Gets the index of the queried value inside the array.

    Args:
        array (array): array to search for the value.
        value (float): value to search for inside the array.

    Returns:
        int: index of the queried value inside the array.
    """
    return np.argmin(np.abs(array - value))


def get_center_val(x, signal_center):
    center_index = get_index(x, signal_center)
    return x[center_index]


def create_fake_data(x, freq_data, sig_x, sig_z, signal_center):
    """
    Create peaks of frequency data from a single signal and an array of frequencies.

    Args:
        x (array): blank x-axis to add the signal to.
        freq_data (array): array of frequencies to create peaks from.
        sig_x (array): x-axis of the signal to create peaks from.
        sig_z (array): power of the signal to create peaks from.
        signal_center (float): center of the signal in hz to create peaks from.

    Returns:
        array: array of power values with peaks created from the signal.
    """
    center_index = get_index(sig_x, signal_center)
    total_power = np.zeros(len(x))

    for frequency in freq_data:
        freq_index = np.argmin(np.abs(x - frequency))
        llim = freq_index - center_index
        ulim = freq_index + len(sig_z) - center_index

        total_power[llim:ulim] += sig_z
    return total_power


def create_fake_elliptical_data(x, freq_data, momentum_data, sig_x, sig_z, sig_center, sig_sigma, rqmap_x, rqmap_z):
    """
    Create peaks of frequency data from a single signal and an array of frequencies.
    Uses momentum data and the provided rqmap to create peak power for each frequency datapoint.

    Args:
        x (array): blank x-axis to add the signal to.
        freq_data (array): array of frequencies.
        momentum_data (array): array of momentum values.
        sig_x (array): x-axis of the signal to create peaks from.
        sig_z (array): power of the signal to create peaks from.
        sig_center (float): center of the signal in hz
        sig_sigma (int): 2*sigma range of the signal will be amplified.
        rqmap_x (array): x-axis of the rqmap.
        rqmap_z (array): power of the rqmap.

    Returns:
        array: array of power values with peaks created from the frequency information.
    """
    center_index = get_index(sig_x, sig_center)
    total_power = np.zeros(len(x))
    sigma = int(2*sig_sigma)

    for i, frequency in enumerate(freq_data):
        freq_index = np.argmin(np.abs(x - frequency))
        llim = freq_index - center_index
        ulim = freq_index + len(sig_z) - center_index

        rqmap_index = get_index(rqmap_x, momentum_data[i])
        power_factor = rqmap_z[rqmap_index]

        amplified_signal = sig_z.copy()
        amplified_signal[center_index-sigma:center_index+sigma] *= power_factor

        # Scale the single_signal by the power_factor in the defined range
        total_power[llim:ulim] += amplified_signal
    return total_power
