import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Classe ou função do filtro KLMS (você deve implementar ou importar esta classe)
class KLMS:
    def __init__(self, step_size, sigma, max_memory):
        self.step_size = step_size
        self.sigma = sigma
        self.max_memory = max_memory
        self.memory = []
        self.weights = []

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def predict(self, x):
        if not self.memory:
            return 0
        return sum(w * self.kernel(x, m) for w, m in zip(self.weights, self.memory))

    def adapt(self, x, d):
        y = self.predict(x)
        e = d - y
        self.memory.append(x)
        self.weights.append(self.step_size * e)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
            self.weights.pop(0)
        return y, e

# Carregar dados do CSV
data = pd.read_csv('Testes com exemplos/sigSp4-P7.csv')
signal = data['Y'].values

# Configurar parâmetros do filtro KLMS
step_size = 0.01
sigma = 0.9
max_memory = 1000

# Inicializar filtro KLMS
klms = KLMS(step_size, sigma, max_memory)

# Aplicar filtro ao sinal
filtered_signal = np.zeros_like(signal)
for i in range(len(signal)):
    x = np.array([signal[i]])  # Entrada atual
    d = signal[i]  # Sinal desejado (nesse caso, igual ao sinal original, pois não temos referência do ruído)
    y, e = klms.adapt(x, d)
    filtered_signal[i] = y

# Visualizar os resultados
plt.figure(figsize=(14, 7))

plt.plot(filtered_signal, label='Sinal Filtrado')
plt.plot(signal, label='Sinal Original')
plt.title('Filtragem Adaptativa KLMS')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
