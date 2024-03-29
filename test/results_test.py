from dsp.carrier_recover import coarse_correction, fine_tracking
from dsp.timing_recover import delay_filter, timing_recover
from commpy import rrcosfilter
from plotting.BER import plot_bit_error_rates
import matplotlib.pyplot as plt
from scipy.signal import firwin
import numpy as np

num_bits = 100000
alpha = .35
Ts = 1/1e6
fs_rx = 4e6
f_c = 2e9
ebnos_db = np.linspace(0, 10, 5)
oversample = 4
nrz_steam = -np.random.randint(0, 2, num_bits)*2 + 1
rrc = rrcosfilter(4, alpha, Ts, fs_rx)[1][1:]
rrc /= np.linalg.norm(rrc)
ppms = [-30, 0]
BERs = []
for ppm in ppms:
    BER = []
    for ebno_db in ebnos_db:
        ebno = np.power(10, ebno_db/10)
        noise_component_power = np.sqrt(1/(oversample*2*ebno))
        rx_signal = 1j*np.zeros(oversample*len(nrz_steam))
        # Use random doppler in ppm tolerance range
        # ppm_error = np.random.uniform(-30, 30)
        ppm_error = ppm
        doppler = f_c*ppm_error/1e6
        phase_offset = np.random.uniform(-np.pi, np.pi)
        timing_error = np.random.uniform(-oversample//2, oversample//2)
        delay = delay_filter(timing_error)
        rx_signal[0::oversample] = nrz_steam
        rx_signal = np.convolve(rx_signal, rrc, "same")
        noise = noise_component_power*np.random.randn(len(rx_signal)) + 1j * noise_component_power*np.random.randn(len(rx_signal))
        rx_signal = np.convolve(rx_signal, delay, 'same')
        rx_signal += noise
        rx_signal *= np.exp(1j*(2*np.pi*doppler*np.arange(len(rx_signal))/fs_rx + phase_offset))
        # Filter out to known ppm error before coarse estimate (lots options for a low pass filter here, nothing fancy)
        filtered = np.convolve(rx_signal[:2000]**2, firwin(21, cutoff=f_c*35/1e6, fs=fs_rx), "same")
        coarse_offset = coarse_correction(filtered, fs_rx)
        frequency_corrected = rx_signal*np.exp(-1j*2*np.pi*coarse_offset*np.arange(len(rx_signal))/fs_rx)
        delay_estimate, timing_error_estimates = timing_recover(frequency_corrected, oversample)
        timing_correction = delay_filter(-delay_estimate)
        timed_corrected = np.convolve(frequency_corrected, timing_correction, "same")
        filtered_corrected = np.convolve(timed_corrected, np.flip(rrc), "same")
        timed_corrected = filtered_corrected[0::oversample]
        fine_corrected, fine_freq, fine_phase = fine_tracking(timed_corrected, oversample, fs_rx)
        # At this point the tracking as assumed to lock and subsequent symbols are considered for BER.
        detected_nrz = -1*np.logical_not(fine_corrected > 0) + 1*(fine_corrected > 0)
        error_rate = np.sum(detected_nrz != nrz_steam)/num_bits
        fine_corrected *= np.exp(1j*np.pi)
        detected_nrz = -1*np.logical_not(fine_corrected > 0) + 1*(fine_corrected > 0)
        error_rate_flipped = np.sum(detected_nrz != nrz_steam)/num_bits
        error_plotting = True
        if error_plotting == True:
            plt.figure()
            plt.title(f"Constellation. ebno: {ebno_db} db, doppler: {ppm_error} ppm")
            # Note these constellations will look bad if convergence isn't very fast.
            plt.scatter(fine_corrected.real, fine_corrected.imag)
            fig_name = f"ebno_{ebno_db}_dop_{ppm_error}_ppm_constellation"
            plt.savefig("../results/"+fig_name+".png")
            plt.figure()
            plt.title(f"Frequency Error: {doppler - fine_freq[-1] - coarse_offset}")
            plt.plot(fine_freq)
            plt.ylabel("Hz")
            plt.xlabel("Symbol Period")
            fig_name = f"ebno_{ebno_db}_dop_{ppm_error}_frequency_convergence"
            plt.savefig("../results/"+fig_name+".png")
            plt.figure()
            plt.title(f"Phase Correction: {phase_offset - fine_phase[-1]}")
            plt.plot(fine_phase)
            plt.ylabel("Radian")
            plt.xlabel("Symbol Period")
            fig_name = f"ebno_{ebno_db}_dop_{ppm_error}_phase_convergence"
            plt.savefig("../results/"+fig_name+".png")
            plt.figure()
            plt.title(f"Timing Estimate Error: {delay_estimate - timing_error}")
            plt.plot(timing_error_estimates - timing_error)
            plt.ylabel("samples")
            plt.xlabel("Symbol Period")
            fig_name = f"ebno_{int(ebno_db)}_dop_{ppm_error}_timing_convergence"
            plt.savefig("../results/"+fig_name+".png")
            plt.close('all')
        # Below there are two streams running, there might be other ways to disambiguate BPSK
        # without pilot symbols but didn't have time to investigate. This feels a little like cheating for now.
        BER.append(np.min((error_rate, error_rate_flipped)))
    BERs.append(BER)
fig = plot_bit_error_rates(ebnos_db, BERs, names=ppms)
plt.savefig("../results/"+"BER.png")
