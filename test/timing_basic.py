from dsp.carrier_recover import coarse_correction, fine_tracking
from dsp.timing_recover import timing_recover, delay_filter, get_delayed_samples
from commpy import rrcosfilter
from plotting.BER import plot_bit_error_rates
import matplotlib.pyplot as plt
from scipy.signal import firwin
import numpy as np

num_bits = 50000
# num_bits = 1000
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

rx_signal = 1j*np.zeros(oversample*len(nrz_steam))
# Use random doppler in ppm tolerance range
ppm_error = np.random.uniform(-30, 30)/1e6
timing_error = np.random.uniform(-oversample, oversample)
timing_error = 2.8967
delay = delay_filter(timing_error)
doppler = f_c*ppm_error
phase_offset = np.random.uniform(-np.pi, np.pi)
symbol_offset = 0
rx_signal[symbol_offset::oversample] = nrz_steam
rx_signal = np.convolve(rx_signal, rrc, "same")
rx_signal_delayed = np.convolve(rx_signal, delay, 'same') # Add timing error
filtered_corrected = np.convolve(rx_signal_delayed, np.flip(rrc), "same")
delay_estimate, timing_error_estimates = timing_recover(filtered_corrected, oversample)
timing_correction = delay_filter(delay_estimate - oversample, delay_filter_length=121)
timed_corrected = np.convolve(filtered_corrected, np.flip(timing_correction), "same")
timed_corrected = timed_corrected[0::oversample]
fine_corrected, fine_freq, fine_phase = fine_tracking(timed_corrected, oversample, fs_rx)
# At this point the tracking as assumed to lock and subsequent symbols are considered for BER.
detected_nrz = -1*np.logical_not(fine_corrected > 0) + 1*(fine_corrected > 0)
error_rate = np.sum(detected_nrz != nrz_steam)/num_bits
print(f"Realtime: {error_rate}")
fine_corrected *= np.exp(1j*np.pi)
detected_nrz = -1*np.logical_not(fine_corrected > 0) + 1*(fine_corrected > 0)
error_rate_flipped = np.sum(detected_nrz != nrz_steam)/num_bits
print(f"Realtime flipped: {error_rate_flipped}")
error_plotting = True
if error_plotting:
    plt.figure()
    plt.title(f"Realtime corrected BER: {error_rate}")
    # Note these constellations will look bad if convergence isn't very fast.
    plt.scatter(fine_corrected.real, fine_corrected.imag)
    plt.figure()
    plt.plot(fine_freq)
    plt.figure()
    plt.title(f"Phase Correction: {phase_offset - fine_phase[-1]}")
    plt.plot(fine_phase)
    plt.figure()
    plt.title(f"Timing error: {timing_error}")
    plt.plot(timing_error_estimates)
print(np.min((error_rate, error_rate_flipped)))
plt.show()