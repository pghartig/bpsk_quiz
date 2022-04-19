import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def plot_bit_error_rates(SNRs_dB, BER_list, names=[], analytic_ser=True):
    """
    Plot BER on log scale with analytical curve for reference.
    :param SNRs_dB:
    :param BER_list:
    :param names:
    :param analytic_ser:
    :return:
    """
    fig = plt.figure()
    for idx, name in enumerate(names):
        plt.plot(SNRs_dB, BER_list[idx], label=f'{name}')
    if analytic_ser==True:
        SNRs_dB = np.linspace(0, 10, 500)
        snrs = np.power(10, SNRs_dB / 10)
        analytic = 1 - norm.cdf(np.sqrt(2*snrs))
        plt.plot(SNRs_dB, analytic, label='Analytic')
    plt.xlabel(r'10log$(E[x]/\sigma^2_n$) [dB]')
    plt.ylabel("SER")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='lower left')
    return fig