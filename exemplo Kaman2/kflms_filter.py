import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Função de kernel gaussiano (RBF)
def gaussian_kernel(x1, x2, sigma=1.0):
    diff = np.array(x1) - np.array(x2)
    return np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))

# Classe do algoritmo KFLMS
class KFLMS:
    def __init__(self, input_size, kernel_func, step_size=0.1, sigma=1.0):
        self.input_size = input_size
        self.kernel_func = kernel_func
        self.step_size = step_size
        self.sigma = sigma
        self.alpha = np.array([])  # Coeficientes alfa
        self.inputs = np.empty((0, input_size))  # Matriz de entradas

    def predict(self, x):
        if len(self.inputs) == 0:
            return 0
        k = np.array([self.kernel_func(x, xi, self.sigma) for xi in self.inputs])
        return np.dot(self.alpha, k)

    def update(self, x, d):
        y = self.predict(x)
        e = d - y
        self.alpha = np.append(self.alpha, self.step_size * e)
        self.inputs = np.vstack([self.inputs, x])
        return y, e

# Função para calcular o PSNR
def calculate_psnr(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Definindo os parâmetros do sinal
fs = 8000
f = 500
n = np.arange(0, 2, 1/fs)

x = sinal(n)
v = gerar_ruido(x, 40)

x_noisy = x + v

# Parâmetros do KLMS
input_size = 1
step_size = 0.1
sigma = 1.0

# Instancia o filtro KFLMS
kflms = KFLMS(input_size, gaussian_kernel, step_size, sigma)

# Treinamento
errors = []
predictions = []
nmse_values = []
psnr_values = []

# Variância do sinal original
variance_x = np.var(x)

# Treinamento com o sinal ruidoso
for i in tqdm(range(len(x_noisy)), desc="Treinamento do KLMS"):
    y, e = kflms.update(np.array([x_noisy[i]]), x[i])
    predictions.append(y)
    errors.append(e)
    
    if i > 0:
        mse = np.mean(np.square(errors))
        nmse = 10 * np.log10(mse / variance_x)
        nmse_values.append(nmse)
        
        psnr = calculate_psnr(np.array(x[:i+1]), np.array(predictions[:i+1]))
        psnr_values.append(psnr)

# Plotando os sinais original e filtrado
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(n, x_noisy, label='Sinal Ruidoso')
plt.plot(n, predictions, label='Sinal Filtrado')
plt.title("Sinal Ruidoso vs Sinal Filtrado")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(n, x, label='Sinal Original')
plt.title("Sinal Original")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(n[1:], nmse_values)
plt.title("Evolução do NMSE")
plt.xlabel("Iteração")
plt.ylabel("NMSE (dB)")

plt.subplot(2, 2, 4)
plt.plot(n[1:], psnr_values)
plt.title("Evolução do PSNR")
plt.xlabel("Iteração")
plt.ylabel("PSNR (dB)")

plt.tight_layout()
plt.savefig('resultados.png')
plt.show()
