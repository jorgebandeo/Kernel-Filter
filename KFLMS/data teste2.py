import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Função de kernel gaussiano (RBF) vetorizada
def gaussian_kernel(x1, x2, sigma=1.0):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.exp(-np.linalg.norm(x1 - x2, axis=1) ** 2 / (2 * sigma ** 2))

# Classe do algoritmo KFLMS com limite de memória
class KFLMS:
    def __init__(self, input_size, kernel_func, step_size=0.1, sigma=1.0, max_memory=1000):
        self.input_size = input_size
        self.kernel_func = kernel_func
        self.step_size = step_size
        self.sigma = sigma
        self.alpha = []  # Coeficientes alfa
        self.inputs = []  # Lista de entradas
        self.max_memory = max_memory  # Limite de memória

    def predict(self, x):
        if len(self.inputs) == 0:
            return 0
        k = self.kernel_func(self.inputs, x, self.sigma)
        return np.dot(self.alpha, k)

    def update(self, x, d, iteration):
        y = self.predict(x)
        e = d - y
        # Adiciona decaimento ao passo de atualização
        step_size = self.step_size / (1 + iteration / 1000)
        self.alpha.append(step_size * e)
        self.inputs.append(x)
        
        # Limitar o tamanho da memória
        if len(self.inputs) > self.max_memory:
            self.inputs.pop(0)
            self.alpha.pop(0)
            
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
df = pd.read_csv('KFxLMS/sigSp4-H4.csv', nrows=3000)

x = df['Y'].values

# Normalizando os dados
x = (x - np.mean(x)) / np.std(x)

# Adicionando ruído ao sinal para simular um ambiente realista (se necessário)
v = np.random.normal(0, 1, len(x)) * np.sqrt(10**(-100/10))
x_noisy = x 

# Parâmetros do KLMS
input_size = 2
step_size = 0.6
sigma = 1.0
max_memory = 1000  # Limite de memória para o kernel

# Instancia o filtro KFLMS
kflms = KFLMS(input_size, gaussian_kernel, step_size, sigma, max_memory)

# Treinamento
errors = []
predictions = []
nmse_values = []
psnr_values = []

# Treinamento com o sinal ruidoso
for i in tqdm(range(len(x_noisy)), desc="Treinamento do KLMS"):
    y, e = kflms.update(np.array([x_noisy[i]]), x[i], i)
    predictions.append(y)
    if i == 0:
        errors.append(0)
        nmse_values.append(0)
        psnr_values.append(0)
    else: 
        errors.append(e)
        mse = np.mean(np.square(errors))
        variance = np.var(x[:i+1])
        nmse = 10 * np.log10(mse / variance)
        nmse_values.append(nmse)
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
plt.plot(predictions, label='Sinal Filtrado')
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
plt.savefig('KFLMS/resultados csv2.png')
plt.show()
