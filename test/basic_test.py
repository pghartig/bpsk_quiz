from dsp.carrier_recover import coarse_correction, fine_tracking
from commpy import rrcosfilter
from plotting.BER import plot_bit_error_rates
import matplotlib.pyplot as plt
from scipy.signal import firwin
import numpy as np

num_bits = 50000
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

for ebno_db in ebnos_db:
    ebno = np.power(10, ebno_db/10)
    noise_component_power = np.sqrt(1/(oversample*2*ebno))
    rx_signal = 1j*np.zeros(oversample*len(nrz_steam))
    # Use random doppler in ppm tolerance range
    ppm_error = np.random.uniform(-30, 30)/1e6
    doppler = f_c*ppm_error
    phase_offset = np.random.uniform(-np.pi, np.pi)
    # Allowed to assume timing is preserved througout.
    timing_offset = 0
    rx_signal[timing_offset::oversample] = nrz_steam
    rx_signal = np.convolve(rx_signal, rrc, "same")
    noise = noise_component_power*np.random.randn(len(rx_signal)) + 1j * noise_component_power*np.random.randn(len(rx_signal))
    rx_signal += noise
    rx_signal *= np.exp(1j*(2*np.pi*doppler*np.arange(len(rx_signal))/fs_rx + phase_offset))
    # Filter out to known ppm error before coarse estimate (lots options for a low pass filter here, nothing fancy)
    filtered = np.convolve(rx_signal[:2000]**2, firwin(21, cutoff=f_c*35/1e6, fs=fs_rx), "same")
    coarse_offset = coarse_correction(filtered, fs_rx)
    frequency_corrected = rx_signal*np.exp(-1j*2*np.pi*coarse_offset*np.arange(len(rx_signal))/fs_rx)
    # Would timing/matched filter would need to be acquired before a full carrier recover (prompt assumes timing).
    # timing_error_estimates = timing_recover(frequency_corrected)
    filtered_corrected = np.convolve(frequency_corrected, np.flip(rrc), "same")
    timed_corrected = filtered_corrected[timing_offset::oversample]
    fine_corrected, fine_freq, fine_phase = fine_tracking(timed_corrected, oversample, fs_rx)
    check = filtered_corrected*np.exp(-1j*2*np.pi*fine_freq[-1]*np.arange(len(rx_signal))/fs_rx)
    check_timed = check[timing_offset::oversample]
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
        plt.title(f"Frequency Error: {doppler - fine_freq[-1] - coarse_offset}")
        plt.plot(fine_freq)
        plt.figure()
        plt.title(f"Phase Correction: {phase_offset - fine_phase[-1]}")
        plt.plot(fine_phase)
    # Below there are two streams running, there might be other ways to disambiguate BPSK
    # without pilot symbols but didn't have time to investigate.
    BER.append(np.min((error_rate, error_rate_flipped)))
plot_bit_error_rates(ebnos_db, [BER], names=["method1"])
plt.show()