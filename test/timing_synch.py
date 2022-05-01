from dsp.carrier_recover import coarse_correction, fine_tracking
from commpy import rrcosfilter
from plotting.BER import plot_bit_error_rates
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

num_bits = 500
alpha = .35
Ts = 1/1e6
fs_rx = 4e6
f_c = 2e9
ebnos_db = np.linspace(0, 10, 3)
oversample = 4
nrz_steam = -np.random.randint(0, 2, num_bits)*2 + 1
BER = []
rrc = rrcosfilter(4, alpha, Ts, fs_rx)[1][1:]
rrc /= np.linalg.norm(rrc)
N = 21

# sample_delay = np.random.uniform(0, 2)
sample_delay = 2.5
delay = np.sinc(np.arange(-N//2, N//2) - sample_delay)
delay *= np.hamming(N)
delay /= np.sum(delay)
delay_correct = np.sinc(np.arange(-N//2, N//2))
delay_correct *= np.hamming(N)
delay_correct /= np.sum(delay_correct)
check = np.exp(1j*2*np.pi*np.arange(200)/20)
check_delay = np.convolve(check, delay, 'same')
check_delay_corrected = np.convolve(check_delay, delay_correct, 'same')
orig_resample = signal.resample_poly(check, 4, 1)
delay_resample = signal.resample_poly(check_delay, 4, 1)
delay_correct_resample = signal.resample_poly(check_delay_corrected, 4, 1)
rx_signal = 1j*np.zeros(oversample*len(nrz_steam))
# Use random doppler in ppm tolerance range
ppm_error = np.random.uniform(-30, 30)/1e6
doppler = f_c*ppm_error
phase_offset = np.random.uniform(-np.pi, np.pi)
# Allowed to assume timing is preserved througout.
timing_offset = 0
rx_signal[timing_offset::oversample] = nrz_steam
rx_signal = np.convolve(rx_signal, rrc, "same")
rx_signal_delayed = np.convolve(rx_signal, delay, "same")
plt.plot(rx_signal_delayed)
plt.plot(rx_signal)
pass