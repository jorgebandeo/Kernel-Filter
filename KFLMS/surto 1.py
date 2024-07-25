import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importando a biblioteca tqdm para a barra de progresso

# Função de kernel gaussiano (RBF)
def gaussian_kernel(x1, x2, sigma=1.0):
    # Calcula o valor do kernel gaussiano entre x1 e x2 com parâmetro sigma
    return np.exp(-np.linalg.norm(np.array(x1) - np.array(x2)) ** 2 / (2 * sigma ** 2))

# Classe do algoritmo KFLMS
class KFLMS:
    def __init__(self, input_size, kernel_func, step_size=0.1, sigma=1.0):
        # Inicializa os parâmetros do KFLMS
        self.input_size = input_size
        self.kernel_func = kernel_func
        self.step_size = step_size
        self.sigma = sigma
        self.alpha = []  # Coeficientes alfa
        self.inputs = []  # Lista de entradas

    def predict(self, x):
        # Faz uma predição usando o filtro KFLMS
        if len(self.inputs) == 0:
            return 0
        # Calcula os valores do kernel para todas as entradas armazenadas
        k = np.array([self.kernel_func(x, xi, self.sigma) for xi in self.inputs])
        # Retorna a predição como o produto escalar dos alfas com os valores do kernel
        return np.dot(self.alpha, k)

    def update(self, x, d):
        # Atualiza os coeficientes do filtro com base no novo dado (x, d)
        y = self.predict(x)  # Prediz o valor atual
        e = d - y  # Calcula o erro
        self.alpha.append(self.step_size * e)  # Atualiza alfa
        self.inputs.append(x)  # Armazena a entrada
        return y, e  # Retorna a predição e o erro

# Função para calcular o PSNR
def calculate_psnr(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    if mse == 0:  # Evitar divisão por zero
        return float('inf')
    max_pixel = np.max(original)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Definindo os parâmetros do sinal
fs = 8000  # Taxa de amostragem
f = 500  # Frequência do sinal senoidal
n = np.arange(0, 1, 1/fs)  # Vetor de tempo para 1 segundos
v = np.random.normal(0, 1, len(n)) * np.sqrt(10**(-40/10))  # Ruído branco gaussiano com SNR de 40 dB

# Definindo o sinal senoidal
x = np.sqrt(2) * np.sin(2 * np.pi * f * n / fs)
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
    y, e = kflms.update(np.array([x_noisy[i]]), x[i])  # Atualiza o filtro com a entrada e saída atuais
    predictions.append(y)  # Armazena a predição
    errors.append(e)  # Armazena o erro
    
    # Calculo do NMSE a partir da segunda iteração
    if i > 0:
        mse = np.mean(np.square(errors))  # Calcula o erro quadrático médio
        variance = np.var(x[:i+1])  # Calcula a variância do sinal original até a iteração atual
        nmse = 10 * np.log10(mse / variance)  # Calcula o NMSE usando a fórmula dada
        nmse_values.append(nmse)  # Armazena o NMSE
    
    # Calculo do PSNR
    if i > 0:
        psnr = calculate_psnr(np.array(x[:i+1]), np.array(predictions[:i+1]))  # Calcula o PSNR
        psnr_values.append(psnr)  # Armazena o PSNR

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

# Plotando a evolução do NMSE
plt.subplot(2, 2, 3)
plt.plot(n[1:], nmse_values)
plt.title("Evolução do NMSE")
plt.xlabel("Iteração")
plt.ylabel("NMSE (dB)")

# Plotando a evolução do PSNR
plt.subplot(2, 2, 4)
plt.plot(n[1:], psnr_values)
plt.title("Evolução do PSNR")
plt.xlabel("Iteração")
plt.ylabel("PSNR (dB)")

plt.tight_layout()
plt.savefig('resultados.png')
plt.show()
