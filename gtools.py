import csv
import bisect
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from lmfit import Model


def main():
    print('gtools is working!')


def time_average(iqdata, freq, t, pwr, t_start, t_stop, dbm=False):
    """ get 1d averaged frequency and power information from 2d mesh """
    # get linear time from time mesh
    time = np.average(t, axis=1)

    # cut time array to specified window
    time_cut = [x for x in np.average(t, axis=1) if t_start < x < t_stop]

    # find indices (location) of t_start and t_stop values
    start_index, end_index = np.where(time == np.min(time_cut))[
        0][0], np.where(time == np.max(time_cut))[0][0]+1

    # get avg freq and power within the time window
    freq_avg = np.average(freq[start_index:end_index], axis=0)

    if dbm:
        pwr_avg = iqdata.get_dbm(np.average(
            pwr[start_index:end_index], axis=0))
    else:
        pwr_avg = np.average(pwr[start_index:end_index], axis=0)
    return freq_avg, pwr_avg


def get_dbm(watt, other=False):
    """ dbm function from iqtools (static method not accessible)"""
    if other:
        watt[watt <= 0] = 10 ** -30
    return 10 * np.log10(np.array(watt) * 1000)


def export_csv(data_dic, filename):
    """ export dictionary of data to csv"""
    headers = data_dic.keys()

    # Make sure all lists in the dictionary have the same length
    num_rows = len(next(iter(data_dic.values())))
    if not all(len(v) == num_rows for v in data_dic.values()):
        raise ValueError(
            "All lists in the data dictionary must have the same length.")

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        # Write rows directly by zipping values
        writer.writerows(dict(zip(headers, row))
                         for row in zip(*data_dic.values()))


def get_bin_range(freq_data, span, center=0):
    """ find start and end bins given span and center """
    start_freq = center - span
    end_freq = center + span

    start_bin = bisect.bisect_left(freq_data, start_freq)
    end_bin = bisect.bisect_right(freq_data, end_freq) - 1

    return start_bin, end_bin


def get_noise_level(amp_data, freq_data, amplitude_thr, win_len=200, polyorder=1, return_mask=False):
    """ get noise level using savgol filter """
    noise_mask = amp_data < amplitude_thr

    filtered_noise = np.copy(amp_data)
    filtered_noise[~noise_mask] = np.nan  # Set non-noise regions to NaN
    filtered_noise[noise_mask] = savgol_filter(
        amp_data[noise_mask], window_length=win_len, polyorder=polyorder)

    valid_indices = np.isfinite(filtered_noise)
    interp_func = interp1d(freq_data[valid_indices], filtered_noise[valid_indices],
                           kind="linear", fill_value="extrapolate")
    fitted_noise = interp_func(freq_data)

    if return_mask:
        return fitted_noise, noise_mask
    else:
        return fitted_noise


def his_fit(data, nbins):
    __, bin_edges = np.histogram(data, bins=nbins)
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def get_center(data, x, amp, cen, sigma, yoffset=0):
    gmodel = Model(gaussian)
    gresult = gmodel.fit(
        data,
        x=x,
        amp=amp,
        cen=cen,
        sigma=sigma,
        yoffset=yoffset
    )
    return gresult.params['cen'].value


def gaussian(x, amp, cen, sigma, yoffset=0):
    return (amp * np.exp(-(x-cen)**2 / (2*sigma**2))) + yoffset


def lorentzian(x, amp, cen, sigma, yoffset):
    return amp * ((sigma**2) / ((x - cen)**2 + sigma**2)) + yoffset


def lorentzian2(x, amp, cen, sigma, yoffset):
    return (amp / (np.pi * sigma)) / (1 + ((x - cen) / sigma) ** 2) + yoffset


def inv_gaussian(x, amp, cen, sigma, yoffset):
    return (amp * -np.exp(-(x-cen)**2 / (2*sigma**2))) + yoffset


def inv_lorentzian(x, amp, cen, sigma, yoffset):
    return -amp * ((sigma**2) / ((x - cen)**2 + sigma**2)) + yoffset


def calculate_dmoq(moq, f, df, gammat=1.395):
    dmoq = -gammat**2 * moq * df/f
    return dmoq


def test():
    return 1+4


if __name__ == '__main__':
    main()
