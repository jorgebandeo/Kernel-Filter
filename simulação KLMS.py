import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parâmetros da simulação
num_samples = 500
n = np.arange(num_samples)

# Sinal de tremor (movimento indesejado)
desired_signal = np.sin(0.1 * np.pi * n)

# Potência do sinal desejado
signal_power = np.mean(desired_signal ** 2)

# SNR desejada em dB
snr_db = 17

# Calcular a potência do ruído necessária para atingir a SNR desejada
noise_power = signal_power / (10 ** (snr_db / 10))

# Gerar ruído com a potência calculada
noise = np.sqrt(noise_power) * np.random.randn(num_samples)

# Sinal de tremor com ruído adicionado
tremor_signal = desired_signal + noise

# Aplicar filtro passa-baixa ao sinal de tremor para reduzir o ruído
def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# filtered_tremor_signal = butter_lowpass_filter(tremor_signal, cutoff=0.2, fs=1.0, order=3)
filtered_tremor_signal = tremor_signal

# Função de kernel gaussiano
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

# Parâmetros do algoritmo KLMS
mu = 0.4  # Taxa de aprendizagem ajustada
sigma = 0.9  # Parâmetro do kernel gaussiano
order = 12 # Ordem do filtro ajustada

# Inicializar vetores
tremor_estimate = np.zeros(num_samples)
cancellation_signal = np.zeros(num_samples)
weights = []

# Algoritmo adaptativo KLMS com kernel gaussiano
for i in range(order, num_samples):
    x_vec = filtered_tremor_signal[i-order:i]  # Vetor de entrada para o filtro
    
    # Calcular a estimativa do tremor usando o kernel gaussiano
    tremor_estimate[i] = sum(w * gaussian_kernel(x_vec, xi, sigma) for (w, xi) in weights)
    
    error = filtered_tremor_signal[i] - tremor_estimate[i]  # Erro entre o tremor real e a estimativa
    
    # Atualização dos pesos do filtro
    weights.append((mu * error, x_vec))
    
    cancellation_signal[i] = -tremor_estimate[i]  # Sinal de cancelamento gerado

# Sinal resultante após o cancelamento de tremor
resulting_signal = filtered_tremor_signal + cancellation_signal

# Ruído remanescente após o cancelamento de tremor
remaining_noise = resulting_signal - desired_signal

# Plotar os sinais incluindo o sinal desejado sem tremor
plt.figure(figsize=(14, 12))

# Sinal de tremor original
plt.subplot(5, 1, 1)
plt.plot(n, tremor_signal, label='Sinal de Tremor Original')
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Sinal de Tremor Original')
plt.legend()
plt.grid(True)

# Sinal de tremor filtrado
plt.subplot(5, 1, 2)
plt.plot(n, filtered_tremor_signal, label='Sinal de Tremor Filtrado')
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Sinal de Tremor Filtrado')
plt.legend()
plt.grid(True)

# Sinal de cancelamento gerado pelo filtro adaptativo
plt.subplot(5, 1, 3)
plt.plot(n, cancellation_signal, label='Sinal de Cancelamento Gerado')
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Sinal de Cancelamento Gerado pelo Filtro Adaptativo')
plt.legend()
plt.grid(True)

# Sinal resultante após o cancelamento de tremor comparado ao sinal desejado sem tremor
plt.subplot(5, 1, 4)
plt.plot(n, desired_signal, label='Sinal Desejado', linestyle='dashed')
plt.plot(n, resulting_signal, label='Sinal Resultante')
plt.plot(n, tremor_estimate, label='tremor ')
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Sinal Resultante Após Cancelamento de Tremor Comparado ao Sinal Desejado')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
