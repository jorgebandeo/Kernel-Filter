import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Classe LMS
class LMS:
    def __init__(self, num_params, learning_step=0.01):
        self.weights = np.zeros(num_params)
        self.learning_step = learning_step

    def predict(self, new_input):
        return np.dot(new_input, self.weights)

    def update(self, new_input, expected):
        prediction = self.predict(new_input)
        error = expected - prediction
        self.weights += self.learning_step * error * new_input
        return error

# Classe Kernel
class Kernel:
    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)

# Classe KLMS
class KLMS(Kernel):
    def __init__(self, num_params, learning_step=0.5, sigma=0.1):
        self.inputs = [np.zeros(num_params)]
        self.weights = [0]
        self.learning_step = learning_step
        self.sigma = sigma
        self.error = None

    def predict(self, new_input):
        estimate = 0
        for i in range(len(self.weights)):
            addition = self.weights[i] * self.kernel(self.inputs[i], new_input)
            estimate += addition
        return estimate

    def update(self, new_input, expected):
        self.error = expected - self.predict(new_input)
        self.inputs.append(new_input)
        new_weights = self.learning_step * self.error
        self.weights.append(new_weights)
        return self.error

# Função principal para processamento e plotagem
def main():
    # Carregar dados do CSV
    file_path = 'Testes com exemplos/sigSp4-P7.csv'  # Substitua pelo caminho do seu arquivo CSV
    data = pd.read_csv(file_path)
    signal = data['Z'].values[:3000]  # Usar apenas as primeiras 3000 linhas

    # Normalizar o sinal
    signal = signal / np.max(np.abs(signal))

    # Inicializar filtros
    lms_filter = LMS(num_params=1, learning_step=0.005)
    klms_filter = KLMS(num_params=1, sigma=0.4, learning_step=0.1)

    # Listas para armazenar erros e sinais filtrados
    lms_errors = []
    klms_errors = []
    lms_output = []
    klms_output = []

    # Processar o sinal com ambos os filtros
    for i in range(1, len(signal)):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # LMS
        lms_error = lms_filter.update(input_signal, expected)
        lms_errors.append(lms_error ** 2)
        lms_output.append(lms_filter.predict(input_signal))

        # KLMS
        klms_error = klms_filter.update(input_signal, expected)
        klms_errors.append(klms_error ** 2)
        klms_output.append(klms_filter.predict(input_signal))

    # Converter MSE para dB
    lms_mse_db = 10 * np.log10(lms_errors)
    klms_mse_db = 10 * np.log10(klms_errors)

    # Calcular MSE médio
    lms_mse = np.mean(lms_errors)
    klms_mse = np.mean(klms_errors)
    
    # Plotar os erros quadráticos médios em dB
    plt.subplot(2, 1, 1)
    plt.plot(lms_mse_db, label='LMS MSE (dB): {:.4f}'.format(10 * np.log10(lms_mse)))
    plt.plot(klms_mse_db, label='KLMS MSE (dB): {:.4f}'.format(10 * np.log10(klms_mse)))
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error (dB)')
    plt.title('Comparison of LMS and KLMS Filters')
    plt.legend()

    # Plotar os sinais filtrados pelos filtros LMS e KLMS
    plt.subplot(2, 1, 2)
    plt.plot(signal[1:], label='Original Signal')
    plt.plot(lms_output, label='LMS Filtered Signal')
    plt.plot(klms_output, label='KLMS Filtered Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.title('Filtered Signals by LMS and KLMS')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./resultados comparativos.png')
    plt.show()

if __name__ == "__main__":
    main()
