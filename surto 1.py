import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import joblib
import os
from tqdm import tqdm

# Define o mesmo ruído para sempre
np.random.seed(42)

# Sinal limpo
def sinal(n):
    return np.sqrt(2) + np.sin((2 * np.pi * 500 * n) / 8000)

# Ruído gaussiano branco
def gerar_ruido(sinal, SNR_db):
    # Calcular potência do sinal
    potencia_sinal = np.mean(sinal ** 2)
    # Converter a potência do sinal para decibéis (dB)
    SNR = 10 ** (SNR_db / 10)
    potencia_ruido = (potencia_sinal/SNR)

    std_dev = np.sqrt(potencia_ruido)
    noise = np.random.normal(0, std_dev, len(sinal))  # tamanho do ruído desejado

    return noise

# Função para aplicar o KLMS em um segmento do sinal
def klms_process_segment(segment, gamma=0.01, n_components=100, learning_rate=0.005):
    n_samples = len(segment)
    X = np.arange(n_samples).reshape(-1, 1)
    
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    X_features = rbf_feature.fit_transform(X)
    
    klms = SGDRegressor(max_iter=1, tol=None, eta0=learning_rate, learning_rate='constant')
    
    normalized_segment = (segment - np.mean(segment)) / np.std(segment)
    
    for i in range(n_samples):
        klms.partial_fit(X_features[i:i+1], normalized_segment[i:i+1])
    
    processed_segment = klms.predict(X_features)
    
    return processed_segment * np.std(segment) + np.mean(segment)

# Função para processar o sinal com janela deslizante e calcular NMSE
def process_signal_with_window(signal, model_path, window_size=100, gamma=0.01, n_components=100, learning_rate=0.005):
    processed_signal = np.zeros_like(signal)
    half_window = window_size // 2
    
    model_exists = os.path.exists(model_path)
    if model_exists:
        klms = joblib.load(model_path)
        print("Modelo carregado do arquivo.")
    else:
        klms = SGDRegressor(max_iter=1, tol=None, eta0=learning_rate, learning_rate='constant')
        print("Novo modelo criado.")
    
    nmse_values = []

    for i in tqdm(range(half_window, len(signal) - half_window), desc=f"Processando {model_path}"):
        segment = signal[i - half_window:i + half_window]
        processed_segment = klms_process_segment(segment, gamma, n_components, learning_rate)
        processed_signal[i - half_window:i + half_window] = processed_segment

        # Calcular o NMSE
        e2_n = (segment - processed_segment) ** 2
        E_e2_n = np.mean(e2_n)
        sigma_d2 = np.var(segment)
        nmse = 10 * np.log10(E_e2_n / sigma_d2)
        nmse_values.append(nmse)
    
    joblib.dump(klms, model_path)
    print("Modelo salvo no arquivo.")
    
    return processed_signal, nmse_values

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

# Processar o sinal com KLMS usando janela deslizante e persistência de modelos
window_size = 100  # Ajuste conforme necessário
processed_signal, nmse_values = process_signal_with_window(signal_with_noise, model_path='klms_model.pkl', window_size=window_size)

# Plotar os resultados
plt.figure(figsize=(12, 18))

plt.subplot(4, 1, 1)
plt.plot(n, signal, label='Sinal Limpo')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal Limpo')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(n, noise, label='Ruído', color='orange')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Ruído 40dB')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(n, signal_with_noise, label='Sinal com Ruído', color='green')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal com Ruído')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(n, processed_signal, label='Sinal Filtrado', color='blue')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal Filtrado')
plt.grid(True)

plt.tight_layout()
plt.savefig('./Sinais Artificias/Simulação.png')
plt.show()

# Plotar o NMSE
plt.figure(figsize=(12, 6))
plt.plot(range(len(nmse_values)), nmse_values, label='NMSE')
plt.xlabel('Iteração')
plt.ylabel('NMSE (dB)')
plt.title('Evolução do NMSE a cada Iteração')
plt.grid(True)
plt.tight_layout()
plt.savefig('./Sinais Artificias/NMSE.png')
plt.show()
