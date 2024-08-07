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

# Parâmetros do algoritmo RLS
order = 12  # Ordem do filtro ajustada
delta = 1.0  # Parâmetro de regularização inicial
lambda_ = 0.99  # Fator de esquecimento

# Inicializar vetores
tremor_estimate = np.zeros(num_samples)
cancellation_signal = np.zeros(num_samples)
P = (1 / delta) * np.eye(order)  # Matriz de covariância inicial
weights = np.zeros(order)  # Pesos do filtro adaptativo

# Algoritmo adaptativo RLS
for i in range(order, num_samples):
    x_vec = filtered_tremor_signal[i-order:i]  # Vetor de entrada para o filtro
    
    y = np.dot(weights, x_vec)  # Estimativa do tremor
    error = filtered_tremor_signal[i] - y  # Erro entre o tremor real e a estimativa

    # Atualização do filtro RLS
    Pi_x = np.dot(P, x_vec)
    gamma = lambda_ + np.dot(x_vec.T, Pi_x)
    K = Pi_x / gamma
    weights += K * error
    P = (P - np.outer(K, Pi_x)) / lambda_
    
    tremor_estimate[i] = y
    cancellation_signal[i] = -y  # Sinal de cancelamento gerado

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
plt.plot(n, tremor_estimate, label='Estimativa do Tremor')
plt.xlabel('Amostras (n)')
plt.ylabel('Amplitude')
plt.title('Sinal Resultante Após Cancelamento de Tremor Comparado ao Sinal Desejado')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
