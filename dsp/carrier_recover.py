import numpy as np
from scipy.signal import firwin
import matplotlib.pyplot as plt

def timing_recover(signal_chunk):
    """
    TODO Would begin with something basic like an early-late algorithm to estimate the derivative
    of the sample point.
    :param signal_chunk:
    :return:
    """
    filter = np.asarray([-1, -1, -1, 0, 1, 1, 1.0])
    filter /= np.sum(np.abs(filter))
    return None

def fine_tracking(input, oversample, fs):
    """
    Recover smaller frequency and phase offset from signal
    :param input: Stream of samples with 1 sample/symbol
    :param oversample: Oversample rate of fs relative to symbol rate
    :param fs: Full rate of original signal (can remove)
    :return: Stream of samples with carrier removed.
    """
    loop_frequency = 0
    loop_phase = 0
    frequency_step = .0001
    phase_step = .0005
    phases = np.zeros(len(input))
    frequencies = np.zeros(len(input))
    phase_errors = np.zeros(len(input))
    error = np.zeros(len(input))
    # Below line assumes input is bpsk so remove modulation with ^2
    input_nl = np.convolve(input**2, firwin(21, .1*fs, fs=fs), "same")
    corrected = 1j*np.zeros(len(input))
    for idx, sample in enumerate(input_nl):
        if idx > 0:
            error[idx] = np.angle(input_nl[idx]*np.conjugate(input_nl[idx-1])) - loop_frequency
            loop_frequency += frequency_step*error[idx]
            phase_errors[idx] = np.angle(sample * np.exp(-1j*(idx*loop_frequency + loop_phase)))
            loop_phase += phase_step*phase_errors[idx]
        phases[idx] = loop_phase
        frequencies[idx] = loop_frequency/2/oversample
        corrected[idx] = input[idx]*np.exp(-1j*(oversample*idx*frequencies[idx] + loop_phase/2))
    return corrected, fs*frequencies/(2*np.pi), phases


def coarse_correction(input, fs):
    """
    Perform a coarse dft based carrier recovery.
    Could increase fft length for high resolution.
    :param input: Note, this assumes that modulation has been removed from input.
    :param fs: Sample rate of incoming signal
    :return: Detected carrier frequency
    """
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(input)))
    # plt.plot(np.linspace(-fs/2, fs/2, len(spectrum)), np.abs(spectrum))
    return ((spectrum.argmax() - len(spectrum)/2)*fs/len(spectrum))/2