import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

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

    def adapt(self, new_input, error):
        y_hat = self.predict(new_input)
        self.update(new_input, error)
        return error

# Função para determinar o ponto de convergência
def find_convergence_point(output, signal, window_size=1):
    errors = np.abs(signal - output)
    average_error = np.mean(errors)
    for i in range(window_size, len(errors)):
        window_error = np.mean(errors[i-window_size:i])
        if window_error < average_error:
            return i
    return len(errors)



# Função principal para processamento e plotagem
def main():
    # Carregar dados do CSV
    file_path = 'Testes com exemplos/sigSp4-P7.csv'  # Substitua pelo caminho do seu arquivo CSV
    data = pd.read_csv(file_path)
    signal = data['Y'].values[:3000]  # Usar apenas as primeiras 3000 linhas

    # Normalizar o sinal
    #signal = signal / np.max(np.abs(signal))

    # Inicializar filtros
    lms_filter = LMS(num_params=1, learning_step=0.005)
    klms_filter = KLMS(num_params=1, sigma=0.9, learning_step=0.9)
    kalman_filter = KalmanFilter(kalman_mse_avg)
    kfxlms_filter = KFxLMS(num_params=1, sigma=0.9, learning_step=0.9)

    # Listas para armazenar sinais filtrados e erros
    lms_output = []
    klms_output = []
    kalman_output = []
    kfxlms_output = []
    lms_errors = []
    klms_errors = []
    kalman_errors = []
    kfxlms_errors = []

    lms_mse_avg = 0
    klms_mse_avg = 0
    kalman_mse_avg = 0
    kfxlms_mse_avg = 0

    convergence_point_lms = 0
    convergence_point_klms = 0
    convergence_point_kalman = 0
    convergence_point_kfxlms = 0

    if(True):
        # Processar o sinal com todos os filtros
        print("Processing with LMS Filter...")
        for i in tqdm(range(1, len(signal)), desc="LMS"):
            input_signal = np.array([signal[i-1]])
            expected = signal[i]

            # LMS
            lms_filter.update(input_signal, expected)
            prediction = lms_filter.predict(input_signal)
            lms_output.append(prediction)
            lms_errors.append((expected - prediction) ** 2)
        convergence_point_lms = find_convergence_point(lms_output, signal[1:])
        lms_mse_avg = np.mean(lms_errors[convergence_point_lms:])
    print("Processing with KLMS Filter...")
    for i in tqdm(range(1, len(signal)), desc="KLMS"):
        input_signal = np.array([signal[i-1]])
        expected = signal[i]

        # KLMS
        klms_filter.update(input_signal, expected)
        prediction = klms_filter.predict(input_signal)
        klms_output.append(prediction)
        klms_errors.append((expected - prediction) ** 2)
    convergence_point_klms = find_convergence_point(klms_output, signal[1:])
    klms_mse_avg = np.mean(klms_errors[convergence_point_klms:])
    if(True):
        print("Processing with Kalman Filter...")
        posteri_estimate = 0.0
        posteri_error_estimate = 1.0
        process_variance = 1e-2
        measurement_variance = 0.01
        for i in tqdm(range(1, len(signal)), desc="Kalman"):
            measurement = signal[i]

            # Predict
            priori_estimate = posteri_estimate
            priori_error_estimate = posteri_error_estimate + process_variance

            # Update
            kalman_gain = priori_error_estimate / (priori_error_estimate + measurement_variance)
            posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
            posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate

            kalman_output.append(posteri_estimate)
            kalman_errors.append((measurement - posteri_estimate) ** 2)
        convergence_point_kalman = find_convergence_point(kalman_output, signal[1:])
        kalman_mse_avg = np.mean(kalman_errors[convergence_point_kalman:])
    if(True):
        print("Processing with KFxLMS Filter...")
        for i in tqdm(range(1, len(signal)), desc="KFxLMS"):
            input_signal = np.array([signal[i-1]])
            expected = signal[i]

            # KFxLMS
            kfxlms_error = kfxlms_filter.adapt(input_signal, expected)
            prediction = kfxlms_filter.predict(input_signal)
            kfxlms_output.append(prediction)
            kfxlms_errors.append((expected - prediction) ** 2)

        # Calcular MSE e ponto de convergência baseado no erro instantâneo
        convergence_point_kfxlms = find_convergence_point(kfxlms_output, signal[1:])
        kfxlms_mse_avg = np.mean(kfxlms_errors[convergence_point_kfxlms:])

    # Definir cores para cada filtro
    colors = {
        'LMS': 'blue',
        'KLMS': 'green',
        'Kalman': 'red',
        'KFxLMS': 'purple'
    }

    # Converter MSE para dB
    lms_mse_db = 10 * np.log10(lms_errors)
    klms_mse_db = 10 * np.log10(klms_errors)
    kalman_mse_db = 10 * np.log10(kalman_errors)
    kfxlms_mse_db = 10 * np.log10(kfxlms_errors)

    # Plotar os erros quadráticos médios em dB ao longo do tempo
    plt.subplot(2, 1, 1)
    plt.plot(lms_mse_db, label=f'LMS MSE (dB) {10 * np.log10(lms_mse_avg):.4f} dB', color=colors['LMS'])
    plt.plot(klms_mse_db, label=f'KLMS MSE (dB) {10 * np.log10(klms_mse_avg):.4f} dB', color=colors['KLMS'])
    plt.plot(kalman_mse_db, label=f'Kalman MSE (dB) {10 * np.log10(kalman_mse_avg):.4f} dB', color=colors['Kalman'])
    plt.plot(kfxlms_mse_db, label=f'KFxLMS MSE (dB) {10 * np.log10(kfxlms_mse_avg):.4f} dB', color=colors['KFxLMS'])
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error (dB)')
    plt.title('MSE of LMS, KLMS, Kalman, and KFxLMS Filters')
    plt.legend()

    # Plotar os sinais filtrados pelos filtros LMS, KLMS, Kalman e KFxLMS
    plt.subplot(2, 1, 2)
    plt.plot(signal[1:], label='Original Signal', color='black')
    plt.plot(lms_output, label='LMS Filtered Signal', color=colors['LMS'])
    plt.plot(klms_output, label='KLMS Filtered Signal', color=colors['KLMS'])
    plt.plot(kalman_output, label='Kalman Filtered Signal', color=colors['Kalman'])
    plt.plot(kfxlms_output, label='KFxLMS Filtered Signal', color=colors['KFxLMS'])
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.title('Filtered Signals by LMS, KLMS, Kalman, and KFxLMS')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./resultados_comparativosY.png')
    #plt.show()

    #Imprimir MSE médio após convergência e pontos de convergência
    print(f"LMS MSE médio após convergência: {10 * np.log10(lms_mse_avg):.4f} dB")
    print(f"KLMS MSE médio após convergência: {10 * np.log10(klms_mse_avg):.4f} dB")
    print(f"Kalman MSE médio após convergência: {10 * np.log10(kalman_mse_avg):.4f} dB")
    print(f"KFxLMS MSE médio após convergência: {10 * np.log10(kfxlms_mse_avg):.4f} dB")

    print(f"Ponto de convergência LMS: {convergence_point_lms}")
    print(f"Ponto de convergência KLMS: {convergence_point_klms}")
    print(f"Ponto de convergência Kalman: {convergence_point_kalman}")
    print(f"Ponto de convergência KFxLMS: {convergence_point_kfxlms}")

if __name__ == "__main__":
    main()
