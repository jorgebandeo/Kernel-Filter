import numpy as np
import matplotlib.pyplot as plt

# Define o mesmo ruído para sempre
np.random.seed(42)

# Sinal limpo
def sinal(n):
    return np.sqrt(2) + np.sin((2 * np.pi * 500 * n) / 8000)

# Ruído gaussiano branco
def gerar_ruido(sinal, SNR_db):
    # Calcular potência do sinal
    potencia_sinal = np.mean(sinal ** 2)
    # Converter a potência do sinal para decibéis (dB
    SNR = 10 ** (SNR_db / 10)
    potencia_ruido = (potencia_sinal/SNR)

    std_dev = np.sqrt(potencia_ruido)
    noise = np.random.normal(0, std_dev, len(sinal))  # tamanho do ruído desejado

    return noise

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



# ////////////////////////////////////////////
    # sinal deve entrar em outra estrutura que vai fazer a KF
    # O resultado sera salvo em imagem mas o procesamento daqui em diante é em outra pasta
    # esse e o Exemplo de uso do elemento
    # posso fazer com que esse codigo receba o sinal filtrado para multrimlas plotagens 
# ////////////////////////////////////////////

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
plt.title('Ruído 40dB')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(n, signal_with_noise, label='Sinal com Ruído', color='green')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.title('Sinal com Ruído')
plt.grid(True)

plt.tight_layout()
plt.savefig('./Sinais Artificias/Simulação.png')
plt.show()
