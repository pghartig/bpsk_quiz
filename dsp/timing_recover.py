import numpy as np
import matplotlib.pyplot as plt

def delay_filter(delay, delay_filter_length=11):
    delay -= 1 #Assume using "same" in convolution
    delay = np.sinc(np.arange(-delay_filter_length//2, delay_filter_length//2) - delay)
    delay *= np.hamming(delay_filter_length)
    delay /= np.sum(delay)
    return delay

def get_delayed_samples(samples, delay):
    d_filter = delay_filter(delay, delay_filter_length=len(samples))
    delayed_samples = np.convolve(samples, d_filter, "same")
    return delayed_samples

def timing_recover(signal_chunk, oversample):
    length_side = oversample - 1
    early_late_filter = np.asarray([-1, -1, -1, 0, 1, 1, 1.0])
    # length_side = oversample - 2
    # early_late_filter = np.asarray([-1, -1, 0, 1, 1.0])
    early_late_filter /= np.sum(np.abs(early_late_filter))
    timing_step = .01    # in units of samples
    delay_estimate = 0
    estimates = []
    check = []
    for symbol_idx in range(int(len(signal_chunk)/oversample - 2)):
        # TODO this could be implemented more efficiently using a Farrow structure
        if symbol_idx > 0:
            # sample_idx = oversample - 1 + symbol_idx*oversample
            sample_idx = symbol_idx*oversample
            # Could also add phase estimate into this stage for increased efficiency.
            samples_mult = signal_chunk[sample_idx-length_side: sample_idx + length_side + 1]
            delayed_samples = get_delayed_samples(samples_mult, -delay_estimate)
            # error = np.sign(np.sum(early_late_filter * delayed_samples))
            error = np.sum(early_late_filter * delayed_samples)
            check.append(error)
            error *= signal_chunk[sample_idx]
            delay_estimate += timing_step*error
            estimates.append(delay_estimate)
    return delay_estimate, np.asarray(estimates)