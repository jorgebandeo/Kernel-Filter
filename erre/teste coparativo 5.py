import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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

# Classe Kalman Filter
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def predict(self):
        self.priori_estimate = self.posteri_estimate
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance

    def update(self, measurement):
        self.kalman_gain = self.priori_error_estimate / (self.priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = self.priori_estimate + self.kalman_gain * (measurement - self.priori_estimate)
        self.posteri_error_estimate = (1 - self.kalman_gain) * self.priori_error_estimate
        return self.posteri_estimate

# Classe KFxLMS
class KFxLMS(Kernel):
    def __init__(self, num_params, learning_step=0.5, sigma=0.1, filter_length=5):
        self.inputs = [np.zeros(num_params)]
        self.weights = [0]
        self.learning_step = learning_step
        self.sigma = sigma
        self.error = None
        self.filter_length = filter_length
        self.secondary_path = np.random.randn(filter_length)  # Simulated secondary path

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

    def adapt(self, new_input, error):
        y_hat = self.predict(new_input)
        secondary_path_output = np.dot(self.secondary_path, np.append(new_input[-self.filter_length+1:], y_hat))
        adapt_error = error - secondary_path_output
        self.update(new_input, adapt_error)
        return adapt_error

# Classe RLS
class RLS:
    def __init__(self, num_params, forgetting_factor=0.99):
        self.weights = np.zeros(num_params)
        self.P = np.eye(num_params) / forgetting_factor
        self.forgetting_factor = forgetting_factor

    def predict(self, new_input):
        return np.dot(new_input, self.weights)

    def update(self, new_input, expected):
        new_input = np.reshape(new_input, (len(new_input), 1))
        expected = np.array([expected])
        y_hat = self.predict(new_input.T)
        error = expected - y_hat
        P_new = self.P - (self.P @ new_input @ new_input.T @ self.P) / (self.forgetting_factor + new_input.T @ self.P @ new_input)
        self.weights += (P_new @ new_input).flatten() * error
        self.P = P_new / self.forgetting_factor
        return error

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
    kalman_filter = KalmanFilter(process_variance=1e-2, measurement_variance=0.01)
    kfxlms_filter = KFxLMS(num_params=1, sigma=0.4, learning_step=0.1)
    rls_filter = RLS(num_params=1, forgetting_factor=0.99)

    # Listas para armazenar erros e sinais filtrados
    lms_errors = []
    klms_errors = []
    kalman_errors = []
    kfxlms_errors = []
    rls_errors = []
    lms_output = []
    klms_output = []
    kalman_output = []
    kfxlms_output = []
    rls_output = []

    # Processar o sinal com todos os filtros
    print("Processing with LMS Filter...")
    for i in tqdm(range(1, len(signal)), desc="LMS"):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # LMS
        lms_error = lms_filter.update(input_signal, expected)
        lms_errors.append(lms_error ** 2)
        lms_output.append(lms_filter.predict(input_signal))

    print("Processing with KLMS Filter...")
    for i in tqdm(range(1, len(signal)), desc="KLMS"):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # KLMS
        klms_error = klms_filter.update(input_signal, expected)
        klms_errors.append(klms_error ** 2)
        klms_output.append(klms_filter.predict(input_signal))

    print("Processing with Kalman Filter...")
    for i in tqdm(range(1, len(signal)), desc="Kalman"):
        expected = signal[i]

        # Kalman Filter
        kalman_filter.predict()
        kalman_output.append(kalman_filter.update(expected))
        kalman_errors.append((expected - kalman_output[-1]) ** 2)

    print("Processing with KFxLMS Filter...")
    for i in tqdm(range(1, len(signal)), desc="KFxLMS"):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # KFxLMS
        kfxlms_error = kfxlms_filter.adapt(input_signal, expected)
        kfxlms_errors.append(kfxlms_error ** 2)
        kfxlms_output.append(kfxlms_filter.predict(input_signal))

    print("Processing with RLS Filter...")
    for i in tqdm(range(1, len(signal)), desc="RLS"):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # RLS
        rls_error = rls_filter.update(input_signal, expected)
        rls_errors.append(rls_error ** 2)
        rls_output.append(rls_filter.predict(input_signal))

    # Converter MSE para dB
    lms_mse_db = 10 * np.log10(lms_errors)
    klms_mse_db = 10 * np.log10(klms_errors)
    kalman_mse_db = 10 * np.log10(kalman_errors)
    kfxlms_mse_db = 10 * np.log10(kfxlms_errors)
    rls_mse_db = 10 * np.log10(rls_errors)

    # Calcular MSE médio
    lms_mse = np.mean(lms_errors)
    klms_mse = np.mean(klms_errors)
    kalman_mse = np.mean(kalman_errors)
    kfxlms_mse = np.mean(kfxlms_errors)
    rls_mse = np.mean(rls_errors)

    # Plotar os erros quadráticos médios em dB
    plt.subplot(2, 1, 1)
    plt.plot(lms_mse_db, label='LMS MSE (dB): {:.4f}'.format(10 * np.log10(lms_mse)))
    plt.plot(klms_mse_db, label='KLMS MSE (dB): {:.4f}'.format(10 * np.log10(klms_mse)))
    plt.plot(kalman_mse_db, label='Kalman MSE (dB): {:.4f}'.format(10 * np.log10(kalman_mse)))
    plt.plot(kfxlms_mse_db, label='KFxLMS MSE (dB): {:.4f}'.format(10 * np.log10(kfxlms_mse)))
    plt.plot(rls_mse_db, label='RLS MSE (dB): {:.4f}'.format(10 * np.log10(rls_mse)))
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error (dB)')
    plt.title('Comparison of LMS, KLMS, Kalman, KFxLMS, and RLS Filters')
    plt.legend()

    # Plotar os sinais filtrados pelos filtros LMS, KLMS, Kalman, KFxLMS e RLS
    plt.subplot(2, 1, 2)
    plt.plot(signal[1:], label='Original Signal')
    plt.plot(lms_output, label='LMS Filtered Signal')
    plt.plot(klms_output, label='KLMS Filtered Signal')
    plt.plot(kalman_output, label='Kalman Filtered Signal')
    plt.plot(kfxlms_output, label='KFxLMS Filtered Signal')
    plt.plot(rls_output, label='RLS Filtered Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.title('Filtered Signals by LMS, KLMS, Kalman, KFxLMS, and RLS')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./resultados_comparativos5.png')
    plt.show()

if __name__ == "__main__":
    main()
