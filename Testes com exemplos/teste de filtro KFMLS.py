import csv
import os
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

# Função para ler o arquivo .txt e extrair os dados
def extrair_dados_txt(arquivo_txt):
    with open(arquivo_txt, 'r') as file:
        linhas = file.readlines()
    
    # Lista para armazenar os dados extraídos
    dados = []

    # Variável para rastrear quando estamos lendo dados numéricos
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

# Função para salvar os dados em um arquivo CSV
def salvar_dados_csv(dados, arquivo_csv):
    with open(arquivo_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])  # Escrever o cabeçalho
        writer.writerows(dados)

# Função para adicionar ruído e filtrar o sinal Y
def adicionar_ruido_e_filtrar(dados, sigma_ruido=1.0):
    y_original = np.array([float(d[1]) for d in dados])
    x = np.arange(len(y_original))

    # Adicionando ruído ao sinal Y
    ruido = np.random.normal(0, 1, len(y_original)) * np.sqrt(10 ** (-40 / 10))
    y_ruidoso = y_original + ruido

    # Parâmetros do KFLMS
    input_size = 1
    step_size = 0.1
    sigma = sigma_ruido

    # Instancia o filtro KFLMS
    kflms = KFLMS(input_size, gaussian_kernel, step_size, sigma)

    # Treinamento
    errors = []
    predictions = []
    nmse_values = []
    psnr_values = []

    for i in tqdm(range(len(y_ruidoso)), desc="Treinamento do KLMS"):
        y_pred, e = kflms.update(np.array([y_ruidoso[i]]), y_original[i])
        predictions.append(y_pred)
        errors.append(e)

        if i > 0:
            mse = np.mean(np.square(errors))
            variance = np.var(y_original[:i + 1])
            nmse = 10 * np.log10(mse / variance)
            nmse_values.append(nmse)

            psnr = calculate_psnr(np.array(y_original[:i + 1]), np.array(predictions[:i + 1]))
            psnr_values.append(psnr)

    # Plotando os resultados
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(y_ruidoso, label='Sinal Ruidoso')
    plt.plot(predictions, label='Sinal Filtrado')
    plt.title("Sinal Ruidoso vs Sinal Filtrado")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(y_original, label='Sinal Original')
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
    plt.savefig('./Testes com exemplos/resultados_kflms.png')
    plt.show()

# Caminho do arquivo .txt a ser processado
arquivo_txt = r'Testes com exemplos/sigSp4-H4.txt'

# Extrair os dados do arquivo .txt
dados = extrair_dados_txt(arquivo_txt)

# Caminho do arquivo .csv de saída
arquivo_csv = r'Testes com exemplos/testesigSp4-H4.csv'

# Salvar os dados no arquivo .csv
salvar_dados_csv(dados, arquivo_csv)

# Adicionar ruído ao sinal Y e filtrar
adicionar_ruido_e_filtrar(dados)

print(f"Dados salvos em {arquivo_csv}")
