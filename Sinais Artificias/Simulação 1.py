import numpy as np
import matplotlib.pyplot as plt

# Define o mesmo ruído para sempre
np.random.seed(42)

# Sinal limpo
def sinal(n):
    return np.sqrt(n) + np.sin((2 * np.pi * 500 * n) / 8000)

# Ruído gaussiano branco
def gerar_ruido(signal, snr_db):
    # Calcular a potência do sinal
    signal_power = np.mean(signal ** 2)
    
    # Calcular a potência do ruído
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Gerar ruído branco gaussiano com a potência desejada
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return noise

# Definir o intervalo e o número de pontos
n = np.linspace(0, 1, 500)

# Gerar o sinal
signal = sinal(n)

# Definir o SNR desejado
snr_db = 20

# Gerar o ruído
noise = gerar_ruido(signal, snr_db)

# Combinar o sinal com o ruído
signal_with_noise = signal + noise

# Plotar os resultados
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(n, signal, label='Sinal Limpo')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal Limpo')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(n, noise, label='Ruído', color='orange')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Ruído')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, signal_with_noise, label='Sinal com Ruído', color='green')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal com Ruído')
plt.grid(True)

plt.tight_layout()
plt.show()
