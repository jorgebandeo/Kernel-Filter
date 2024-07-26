import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

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
            x, y, z, w, r, d = valores[0], valores[1], valores[2], valores[3], valores[4], valores[5]
            dados.append([x, y, z, w, r, d])
    
    return dados

# Função para salvar os dados em um arquivo CSV
def salvar_dados_csv(dados, arquivo_csv):
    with open(arquivo_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z","W", "R", "D"])  # Escrever o cabeçalho
        writer.writerows(dados)

# Função para plotar os dados de um arquivo CSV
def plotar_dados_csv(arquivo_csv):
    # Ler o arquivo CSV usando pandas
    df = pd.read_csv(arquivo_csv)

    # Plotar os dados
    plt.figure(figsize=(10, 6))
    plt.plot(df["X"], label="X")
    plt.plot(df["Y"], label="Y")
    plt.plot(df["Z"], label="Z")
    plt.plot(df["W"], label="W")
    plt.plot(df["R"], label="R")
    plt.plot(df["D"], label="D")    
    plt.xlabel("Índice")
    plt.ylabel("Valores")
    plt.title("Plotagem dos Dados de X, Y e Z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./Testes com exemplos/normal.png')
    plt.show()

# Caminho do arquivo .txt a ser processado
arquivo_txt = r'Testes com exemplos/sigSp4-H4.txt'

# Extrair os dados do arquivo .txt
dados = extrair_dados_txt(arquivo_txt)

# Caminho do arquivo .csv de saída
arquivo_csv = r'Testes com exemplos/sigSp4-H4.csv'

# Salvar os dados no arquivo .csv
salvar_dados_csv(dados, arquivo_csv)

# Plotar os dados do arquivo .csv
plotar_dados_csv(arquivo_csv)

print(f"Dados salvos em {arquivo_csv}")
