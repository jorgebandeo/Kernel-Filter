import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Função de kernel gaussiano (RBF)
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(np.array(x1) - np.array(x2)) ** 2 / (2 * sigma ** 2))

# Classe do algoritmo KFLMS
class KFLMS:
    def __init__(self, input_size, kernel_func, step_size=0.1, sigma=1.0):
        self.input_size = input_size
        self.kernel_func = kernel_func
        self.step_size = step_size
        self.sigma = sigma
        self.alpha = []
        self.inputs = []

    def predict(self, x):
        if len(self.inputs) == 0:
            return 0
        k = np.array([self.kernel_func(x, xi, self.sigma) for xi in self.inputs])
        return np.dot(self.alpha, k)

    def update(self, x, d):
        y = self.predict(x)
        e = d - y
        self.alpha.append(self.step_size * e)
        self.inputs.append(x)
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
fs = 1000
f = 50
n = np.arange(0, 1, 1/fs)
sinal_puro = np.sin(2 * np.pi * f * n)

# Gerar diferentes tipos de ruídos
np.random.seed(0)

# Ruído Branco
white_noise = sinal_puro + np.random.normal(0, 1, len(n))

# Ruído Gaussiano
gaussian_noise = sinal_puro + np.random.normal(0, 0.5, len(n))

# Ruído de Banda Limitada (100-200 Hz)
band_limited_noise = sinal_puro + np.sin(2 * np.pi * 150 * n) * np.random.normal(0, 0.5, len(n))

# Ruído Impulsivo
impulse_noise = sinal_puro.copy()
impulse_indices = np.random.randint(0, len(n), 50)
impulse_noise[impulse_indices] += np.random.normal(0, 5, 50)

# Ruído Rosa (aproximado)
pink_noise = sinal_puro + np.cumsum(np.random.normal(0, 1, len(n)))

# Ruído de Fundo
background_noise = sinal_puro + np.random.normal(0, 0.2, len(n)) + np.sin(2 * np.pi * 50 * n) * 0.2

# Lista de ruídos
ruidos = [white_noise, gaussian_noise, band_limited_noise, impulse_noise, pink_noise, background_noise]
ruidos_nomes = ['Ruído Branco', 'Ruído Gaussiano', 'Ruído de Banda Limitada', 'Ruído Impulsivo', 'Ruído Rosa', 'Ruído de Fundo']

# Parâmetros do KLMS
input_size = 1
step_size = 0.1
sigma = 1.0

# Aplicar KLMS para cada tipo de ruído
plt.figure(figsize=(18, 18))

for i, (ruido, nome) in enumerate(zip(ruidos, ruidos_nomes)):
    kflms = KFLMS(input_size, gaussian_kernel, step_size, sigma)
    predictions = []
    errors = []
    nmse_values = []
    psnr_values = []
    
    for j in tqdm(range(len(ruido)), desc=f"KLMS para {nome}"):
        y, e = kflms.update(np.array([ruido[j]]), sinal_puro[j])
        predictions.append(y)
        errors.append(e)

        if j > 0:
            mse = np.mean(np.square(errors))
            variance = np.var(sinal_puro[:j+1])
            nmse = 10 * np.log10(mse / variance)
            nmse_values.append(nmse)
            psnr = calculate_psnr(np.array(sinal_puro[:j+1]), np.array(predictions[:j+1]))
            psnr_values.append(psnr)
    
    # Plotar os resultados
    plt.subplot(6, 4, i*4 + 1)
    plt.plot(n, ruido, label='Sinal Ruidoso')
    plt.plot(n, predictions, label='Sinal Filtrado')
    plt.title(f"Sinal Ruidoso vs Sinal Filtrado ({nome})")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(6, 4, i*4 + 2)
    plt.plot(n, sinal_puro, label='Sinal Original')
    plt.title("Sinal Original")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(6, 4, i*4 + 3)
    plt.plot(n[1:], nmse_values)
    plt.title("Evolução do NMSE")
    plt.xlabel("Iteração")
    plt.ylabel("NMSE (dB)")

    plt.subplot(6, 4, i*4 + 4)
    plt.plot(n[1:], psnr_values)
    plt.title("Evolução do PSNR")
    plt.xlabel("Iteração")
    plt.ylabel("PSNR (dB)")

plt.tight_layout()
plt.savefig('teste de ruidos/teste de ruidos.png')
plt.show()
