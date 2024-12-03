import numpy as np
import iqtools as iq
import FakeDataMaker as fm
import gtools as gt
from lmfit import Model

# import data
rawdata = 'data/fumikara.TIQ'
iqdata = iq.get_iq_object(rawdata)

# set params & read data
lframes = int(2000)  # number of bins in freq.
freq_bin_size = iqdata.fs/lframes
t_bin_size = 1/freq_bin_size

nframes = int(iqdata.nsamples_total/lframes)
iqdata.read(nframes, lframes)

# get power spectrum
iqdata.window = 'hamming'
iqdata.method = 'mtm'
xx, yy, zz = iqdata.get_power_spectrogram(nframes=nframes, lframes=lframes)

xxcut, yycut, zzcut = iq.get_cut_spectrogram(xx, yy, zz, ycen=0, yspan=3.6*2)

# get averaged spectrogram
freq_avg, pwr_avg = gt.time_average(
    iqdata, xxcut, yycut, zzcut, 0, 60, dbm=False)

gmodel = Model(fm.gaussian)
gresult_single_signal_fit = gmodel.fit(
    pwr_avg,
    x=freq_avg,
    amp=4e-6,
    cen=freq_avg[np.argmax(pwr_avg)],
    sigma=500,
    yoffset=3e-6,
)

current_res = int(-(-(np.max(freq_avg) - np.min(freq_avg)) // len(freq_avg)))
intp_factor = current_res
xnew = np.linspace(np.min(freq_avg), np.max(
    freq_avg), len(freq_avg)*intp_factor)
interp_signal = np.interp(xnew, freq_avg, pwr_avg)

with open("data/interp_single_sig.csv", "w") as file:
    for x, y in zip(xnew, interp_signal):
        file.write(f"{x}, {y}\n")
