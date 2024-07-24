import numpy as np
from Algoritm import run_kalman_filter, plot

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
    potencia_ruido = (potencia_sinal / SNR)

    std_dev = np.sqrt(potencia_ruido)
    noise = np.random.normal(0, std_dev, len(sinal))  # tamanho do ruído desejado

    return noise

if __name__ == '__main__':
    numStates = 500
    
    # Definir o intervalo e o número de pontos
    n = np.linspace(0, 1, numStates)
    
    # Gerar o sinal
    signal = sinal(n)
    
    # Definir o SNR desejado
    snr_db = 40
    
    # Gerar o ruído
    noise = gerar_ruido(signal, snr_db)
    
    # Combinar o sinal com o ruído
    signal_with_noise = signal + noise
    
    # Executar o filtro de Kalman
    
    filtered, nmse_values, mse = run_kalman_filter(signal, signal_with_noise, numStates)

    plot(signal, signal_with_noise, filtered, nmse_values)
    print("Filtered signal:", filtered)
    print("NMSE values:", nmse_values)
    print("Final Mean Squared Error:", mse)
