import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import time

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

    return seq[10:]  # Ajuste este valor conforme necessário

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

def plot(original, noisy, filtered, nmse_values, processing_times):
    filtered = filtered[40:]
    min_length = min(len(original), len(noisy), len(filtered))
    original = original[:min_length]
    noisy = noisy[:min_length]
    filtered = filtered[:min_length]
    
    x = range(min_length)
    
    fig = plt.figure(figsize=(14, 10))
    plt.style.use('dark_background')
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, noisy, color='#7116bc', linestyle='--', label='Noisy Signal')
    ax1.plot(x, filtered, color='#FF00FF', label='Filtered Signal')
    ax1.plot(x, original, color='#00FF00', label='Original Signal')
    ax1.set_xlabel('Time', color='white')
    ax1.set_ylabel('Value', color='white')
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(color='#444444', linestyle='--')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(range(len(nmse_values)), nmse_values, color='#FFA500', label='NMSE')
    ax2.set_xlabel('Time', color='white')
    ax2.set_ylabel('NMSE (dB)', color='white')
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(color='#444444', linestyle='--')

    # Add a new subplot for processing times
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(range(len(processing_times)), processing_times, color='#00BFFF', label='Processing Time per Sample')
    ax3.axhline(y=np.mean(processing_times), color='r', linestyle='--', label='Average Processing Time')
    ax3.set_xlabel('Sample Index', color='white')
    ax3.set_ylabel('Time (s)', color='white')
    ax3.legend(loc='upper right', frameon=False)
    ax3.grid(color='#444444', linestyle='--')

    plt.tight_layout()
    plt.savefig('resultados_Fibonacci.png', facecolor='#1e1e1e')
    plt.show()

def kfilter(rw, numStates, seq, iterations=1):
    filtered = np.zeros_like(rw)
    nmse_values = []
    processing_times = np.zeros_like(rw)
    sigma_d_squared = np.var(rw)

    for iteration in range(iterations):
        if iteration > 0:
            rw = filtered.copy()
        
        for i in range(numStates, len(rw)):
            start_time = time.time()
            
            port = portion(seq, numStates - 1)
            est = estimate(port, rw[i - numStates:i], numStates - 1)
            filtered[i] = est
            


            if iteration == iterations - 1:
                mse = np.mean((filtered[numStates:i + 1] - rw[numStates:i + 1])**2)
                nmse = 10 * np.log10(mse / sigma_d_squared)
                nmse_values.append(nmse)

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times[i]= processing_time + processing_times[i]

    for i in range(len(processing_times)):
        processing_times[i] = processing_times[i] / iterations
    filtered = np.clip(filtered, np.min(rw), np.max(rw))
    return filtered, nmse_values, processing_times

def calculate_mse(filtered, rw, numStates):
    diff_list = np.square(np.subtract(filtered[numStates:], rw[numStates:]))
    mse = float(sum(diff_list)) / float(len(diff_list))
    print("The mean squared error is ", mse)
    return mse

def run_kalman_filter(noisy_signal, numStates):
    seq = fibonacci(2000)
    filtered, nmse_values, processing_times = kfilter(noisy_signal, numStates, seq, iterations=20)
    mse = calculate_mse(filtered, noisy_signal, numStates)
    
    return filtered, nmse_values, processing_times, mse

def extrair_dados_txt(arquivo_txt):
    with open(arquivo_txt, 'r') as file:
        linhas = file.readlines()
    
    dados = []
    lendo_dados = False

    for linha in linhas:
        linha = linha.strip()
        if linha == "#</meta>":
            lendo_dados = True
            continue
        if lendo_dados and linha:
            valores = linha.split()
            x, y, z = valores[0], valores[1], valores[2]
            dados.append([x, y, z])
    
    return dados

def salvar_dados_csv(dados, arquivo_csv):
    with open(arquivo_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])
        writer.writerows(dados)

def adicionar_ruido_e_filtrar(dados, numStates=50):
    y_original = np.array([float(d[2]) for d in dados])
    x = np.arange(len(y_original))

    ruido = np.random.normal(0, 1, len(y_original)) * np.sqrt(10 ** (4 / 10))
    y_ruidoso = y_original + ruido

    filtered, nmse_values, processing_times, mse = run_kalman_filter(y_ruidoso, numStates)

    nmse_values = nmse_values[:len(filtered) - numStates]

    plot(y_original[numStates:], y_ruidoso[numStates:], filtered[numStates:], nmse_values, processing_times)

arquivo_txt = r'Testes com exemplos/sigSp4-H4.txt'
dados = extrair_dados_txt(arquivo_txt)
arquivo_csv = r'Testes com exemplos/Fibonacci_sigSp4-H4.csv'
salvar_dados_csv(dados, arquivo_csv)
adicionar_ruido_e_filtrar(dados)

print(f"Dados salvos em {arquivo_csv}")
