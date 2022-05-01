import numpy as np
import matplotlib.pyplot as plt

def delay_filter(delay, delay_filter_length=11):
    delay = np.sinc(np.arange(-delay_filter_length//2, delay_filter_length//2) - delay)
    delay *= np.hamming(delay_filter_length)
    delay /= np.sum(delay)
    return delay

def get_delay_filter_coeffiecients():
    return None

def get_delayed_samples(samples, delay):
    d_filter = delay_filter(delay, delay_filter_length=len(samples))
    delayed_samples = np.convolve(samples, d_filter, "same")
    return delayed_samples

def timing_recover(signal_chunk, oversample):
    length_side = oversample - 1
    early_late_filter = np.asarray([1, 1, 1, 0, 1, 1, 1.0])
    # filter = np.asarray([1, 1, 0, -1, -1.0])
    early_late_filter /= np.sum(np.abs(early_late_filter))
    timing_step = .05    # in units of samples
    delay_estimate = 0
    estimtes = []
    for symbol_idx in range(int(len(signal_chunk)/oversample)):
        # TODO this could be implemented more efficiently using a Farrow structure
        sample_idx = oversample - 1 + symbol_idx*oversample
        delayed_samples = get_delayed_samples(signal_chunk[sample_idx-length_side: sample_idx + length_side + 1], delay_estimate)
        error = np.sign(np.sum(early_late_filter * delayed_samples))
        error *= np.sign(signal_chunk[sample_idx])
        delay_estimate += timing_step*error
        estimtes.append(delay_estimate)
        # Adjust estimated offset using error and step size
    return delay_estimate, estimtes
