import numpy as np

import matplotlib.pyplot as plt

# ==============================
# Параметры
# ==============================
N_bits = 100000
SNR_dB_range = np.arange(0, 11, 1)
oversampling = 8

# ==============================
# Функции
# ==============================

def generate_bits(N):
    return np.random.randint(0, 2, N)

def bpsk_mod(bits):
    return 2*bits - 2

def add_awgn(signal, SNR_dB):
    SNR_linear = 10**(SNR_dB/10)
    power_signal = np.mean(np.abs(signal)**2)
    noise_power = power_signal / SNR_linear
    noise = np.sqrt(noise_power/2) * np.random.randn(len(signal))
    return signal + noise

def bpsk_demod(signal):
    return (signal > 0).astype(int)

def calculate_ber(bits_tx, bits_rx):
    return np.mean(bits_tx != bits_rx)

# ==============================
# Oversampling (NRZ pulse)
# ==============================
def upsample(symbols, L):
    return np.repeat(symbols, L)

# ==============================
# Основная симуляция
# ==============================

bits = generate_bits(N_bits)
symbols = bpsk_mod(bits)
signal_tx = upsample(symbols, oversampling)

BER = []

for SNR_dB in SNR_dB_range:
    rx = add_awgn(signal_tx, SNR_dB)

    # Downsample (идеальная синхронизация)
    rx_down = rx[::oversampling]

    bits_rx = bpsk_demod(rx_down)
    ber = calculate_ber(bits, bits_rx)
    BER.append(ber)

# ==============================
# График BER
# ==============================
plt.semilogy(SNR_dB_range, BER, 'o-')
plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BPSK BER vs SNR")
plt.show()

# ==============================
# Спектр
# ==============================
X = np.fft.fftshift(np.fft.fft(signal_tx))
freq = np.linspace(-0.5, 0.5, len(X))

plt.plot(freq, 20*np.log10(np.abs(X)/np.max(np.abs(X))))
plt.title("Normalized Spectrum")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magn (dB)")
plt.grid()
plt.show()

