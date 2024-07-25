import numpy as np
import pandas as pd
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
        self.alpha = []  # Coeficientes alfa
        self.inputs = []  # Lista de entradas

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

# Leitura dos dados do CSV
df = pd.read_csv('KF com Peso Fibonacci/teste com dados/Dataset.csv')
x = df['aX'].values

# Adicionando ruído ao sinal para simular um ambiente realista (se necessário)
v = np.random.normal(0, 1, len(x)) * np.sqrt(10**(-40/10))
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

# Treinamento com o sinal ruidoso
for i in tqdm(range(len(x_noisy)), desc="Treinamento do KLMS"):
    y, e = kflms.update(np.array([x_noisy[i]]), x[i])
    predictions.append(y)
    errors.append(e)
    
    if i > 0:
        mse = np.mean(np.square(errors))
        variance = np.var(x[:i+1])
        nmse = 10 * np.log10(mse / variance)
        nmse_values.append(nmse)
    
    if i > 0:
        psnr = calculate_psnr(np.array(x[:i+1]), np.array(predictions[:i+1]))
        psnr_values.append(psnr)

# Plotando os sinais original e filtrado
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(x_noisy, label='Sinal Ruidoso')
plt.plot(predictions, label='Sinal Filtrado')
plt.title("Sinal Ruidoso vs Sinal Filtrado")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, label='Sinal Original')
plt.title("Sinal Original")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.legend()

# Plotando a evolução do NMSE
plt.subplot(2, 2, 3)
plt.plot(nmse_values)
plt.title("Evolução do NMSE")
plt.xlabel("Iteração")
plt.ylabel("NMSE (dB)")

# Plotando a evolução do PSNR
plt.subplot(2, 2, 4)
plt.plot(psnr_values)
plt.title("Evolução do PSNR")
plt.xlabel("Iteração")
plt.ylabel("PSNR (dB)")

plt.tight_layout()
plt.savefig('KFLMS/resultados csv.png')
plt.show()
