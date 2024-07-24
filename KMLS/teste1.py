import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# Sinal limpo
def sinal(n):
    return np.sqrt(2) + np.sin((2 * np.pi * 500 * n) / 8000)

# Ruído gaussiano branco
def gerar_ruido(sinal, SNR_db):
    potencia_sinal = np.mean(sinal ** 2)
    SNR = 10 ** (SNR_db / 10)
    potencia_ruido = (potencia_sinal / SNR)
    std_dev = np.sqrt(potencia_ruido)
    noise = np.random.normal(0, std_dev, len(sinal))
    return noise

# KLMS parameters
eta = 0.5  # Learning rate
sigma = 0.1  # Kernel width
n_iter = 100  # Number of iterations

# Definir o intervalo e o número de pontos
n = np.linspace(0, 1, 500)

# Gerar o sinal
signal = sinal(n)

# Definir o SNR desejado
snr_db = 40

# Gerar o ruído
noise = gerar_ruido(signal, snr_db)

# Combinar o sinal com o ruído
signal_with_noise = signal + noise

# Kernel LMS algorithm
N = len(signal_with_noise)
K = rbf_kernel(signal_with_noise.reshape(-1, 1), gamma=1/(2*sigma**2))
y = signal_with_noise
d = signal

alpha = np.zeros(N)

for k in range(N):
    y_hat = np.dot(alpha, K[:, k])
    e = d[k] - y_hat
    alpha[k] = eta * e

output_klms = np.dot(alpha, K)

# Plotting results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(n, signal, label='Original Signal')
plt.plot(n, signal_with_noise, label='Noisy Signal')
plt.plot(n, output_klms, label='KLMS Output')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('KLMS Algorithm')
plt.show()
