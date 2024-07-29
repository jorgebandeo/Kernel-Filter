import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ler o CSV
data = pd.read_csv('Testes com exemplos/sigSp4-H4.csv')  # Substitua pelo caminho correto
y_data = data['Y'].values

# Normalizar os dados
y_data_mean = np.mean(y_data)
y_data_std = np.std(y_data)
y_data_normalized = (y_data - y_data_mean) / y_data_std

# Função para criar janelas de entrada e saída
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        X.append(a)
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 10
X, y = create_dataset(y_data_normalized, look_back)

# Implementação do algoritmo LMS com plotagem em tempo real
def lms_predict_real_time(X, y, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    mse = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='MSE (dB)')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('MSE (dB)')
    ax.set_xlim(0, epochs)
    ax.set_ylim(-100, 10)
    ax.legend()
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(n_samples):
            y_pred = np.dot(X[i], weights) + bias
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error
            total_error += error**2
        mse.append(total_error / n_samples)
        
        # Atualizar o gráfico
        mse_db = 10 * np.log10(mse)
        line.set_xdata(np.arange(len(mse_db)))
        line.set_ydata(mse_db)
        ax.draw_artist(ax.patch)
        ax.draw_artist(line)
        fig.canvas.flush_events()
    
    plt.ioff()
    return weights, bias, mse

# Treinando o modelo LMS com plotagem em tempo real
weights, bias, mse = lms_predict_real_time(X, y, lr=0.01, epochs=100)

# Fazendo previsões
y_pred = np.dot(X, weights) + bias

# Desnormalizando as previsões
y_pred_denorm = y_pred * y_data_std + y_data_mean

# Plotando os resultados finais
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(y_data[look_back:], label='Verdadeiro')
plt.plot(y_pred_denorm, label='Previsto')
plt.xlabel('Tempo')
plt.ylabel('Y')
plt.legend()

# Calculando e plotando o MSE final em dB
mse_db = 10 * np.log10(mse)
plt.subplot(2, 1, 2)
plt.plot(mse_db)
plt.xlabel('Épocas')
plt.ylabel('MSE (dB)')
plt.title('Erro Quadrático Médio (MSE) em dB')

plt.tight_layout()
plt.show()

# Utilizando os dados mais recentes para previsão
recent_data = y_data_normalized[-look_back:]
future_predict = np.dot(recent_data, weights) + bias
future_predict_denorm = future_predict * y_data_std + y_data_mean

print("Previsão futura: ", future_predict_denorm)
