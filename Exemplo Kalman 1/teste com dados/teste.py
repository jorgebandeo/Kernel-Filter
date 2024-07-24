import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fibonacci(n):
    f_0 = 0
    f_1 = 1
    seq = [f_0, f_1]
    curr = 0 
    l1 = f_0
    l2 = f_1
    i = 2 
    while i < n:
        curr = l1 + l2
        seq.append(curr)
        l1 = seq[i - 1]
        l2 = curr 
        i += 1

    return seq

def portion(seq, stateNum):
    weight = [] 
    start_idx = stateNum * 2 + 1 
    curr_idx = start_idx
    dom = seq[start_idx + 1]
    numTerms = stateNum + 1 
    while numTerms != 0:
        weight.append(float(seq[curr_idx]))
        curr_idx -= 2
        numTerms -= 1

    return np.divide(weight, dom)

def estimate(port, rw, stateNum):
    port_sec = np.array(port)
    rw_sec = np.array(rw[:stateNum + 1])
    rw_sec = np.array(list(reversed(rw_sec)))
    est = np.dot(port_sec, rw_sec.T)
    return est

def plot(original, filtered, nmse_values):
    x = range(1, len(filtered) + 1)
    
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x, original[:len(filtered)], color='b', linestyle='--', label='Noisy Signal')
    plt.plot(x, filtered, color='r', label='Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, nmse_values, color='g', label='NMSE')
    plt.xlabel('Time')
    plt.ylabel('NMSE (dB)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def kfilter(rw, numStates, seq):
    filtered = np.zeros_like(rw)
    nmse_values = []
    sigma_d_squared = np.var(rw)
    for i in range(numStates, len(rw)):
        port = portion(seq, numStates - 1)
        est = estimate(port, rw[i - numStates:i], numStates - 1)
        filtered[i] = est
        mse = np.mean((filtered[numStates:i + 1] - rw[numStates:i + 1])**2)
        nmse = 10 * np.log10(mse / sigma_d_squared)
        nmse_values.append(nmse)

    return filtered, nmse_values

def calculate_mse(filtered, rw, numStates):
    diff_list = np.square(np.subtract(filtered[numStates:], rw[numStates:]))
    mse = float(sum(diff_list)) / float(len(diff_list))
    print("The mean squared error is ", mse)
    return mse

def run_kalman_filter(noisy_signal, numStates):
    seq = fibonacci(2000)
    filtered, nmse_values = kfilter(noisy_signal, numStates, seq)
    mse = calculate_mse(filtered, noisy_signal, numStates)
    plot(noisy_signal[numStates:], filtered[numStates:], nmse_values)
    return filtered, nmse_values, mse

# Carregar o arquivo CSV
file_path = 'Exemplo Kalman 1/teste com dados/Dataset.csv'  # Coloque o caminho do arquivo CSV aqui
data = pd.read_csv(file_path)

# Extrair a primeira coluna (aX) como o sinal ruidoso
noisy_signal = data['aX'].values

# Aplicação do filtro
numStates = 20 # Número de estados para o filtro
filtered_signal, nmse_values, mse = run_kalman_filter(noisy_signal, numStates)


