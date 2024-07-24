import numpy as np

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

# KFxLMS parameters
eta = 0.5  # Learning rate
sigma = 0.1  # Kernel width
N = 500  # Number of iterations

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

# Desired signal is zero in ANC application
desired_signal = np.zeros(N)

# KFxLMS algorithm
alpha = np.zeros(N)
y_hat = np.zeros(N)
error = np.zeros(N)
output_kfxlms = np.zeros(N)

# Kernel function
def kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

# Filtered-x LMS with kernel
for k in range(1, N):
    phi = np.array([kernel(signal_with_noise[i], signal_with_noise[k], sigma) for i in range(k)])
    y_hat[k] = np.dot(alpha[:k], phi)
    error[k] = desired_signal[k] - y_hat[k]
    alpha[k] = eta * error[k]

    output_kfxlms[k] = y_hat[k]

# Plotting results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(n, signal_with_noise, label='Noisy Signal')
plt.plot(n, output_kfxlms, label='KFxLMS Output')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('KFxLMS Algorithm')
plt.show()
