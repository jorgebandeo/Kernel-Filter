### README.md

# Implementação do Filtro de Kalman com Peso Fibonacci

Este repositório contém uma implementação de um Filtro de Kalman utilizando uma sequência de Fibonacci para ponderação dos estados. O código fornece uma abordagem matemática e científica para estimativa de sinal e redução de ruído.

## Visão Geral

O código fornecido implementa um Filtro de Kalman usando uma sequência de Fibonacci para determinar os pesos para a estimativa de estados. O filtro processa um sinal ruidoso e produz uma saída filtrada, estimando o sinal original. O desempenho do filtro é avaliado usando o Erro Médio Quadrático Normalizado (NMSE).

## Análise Matemática

### Geração da Sequência de Fibonacci

A função `fibonacci(n)` gera os primeiros `n` números da sequência de Fibonacci. A sequência é usada para calcular os pesos do filtro. A sequência de Fibonacci é definida pela fórmula:

```math
F(n) = F(n-1) + F(n-2)
```

com $F(0) = 0$ e $F(1) = 1 $.

```python
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
```

### Cálculo dos Pesos

A função `portion(seq, stateNum)` calcula os pesos para o filtro com base na sequência de Fibonacci. Os pesos são normalizados pela divisão pelo termo da sequência no índice `stateNum + 1`. A fórmula é:

```math
W(i) = \frac{F(2i + 1)}{F(2 \times stateNum + 2)}
```

```python
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
```

### Estimativa de Estado

A função `estimate(port, rw, stateNum)` estima o estado do sinal usando os pesos calculados. A estimativa é a soma ponderada dos estados reversos do sinal:

```math
\hat{x} = \sum_{i=0}^{stateNum} W(i) \cdot rw[stateNum - i]
```

```python
def estimate(port, rw, stateNum):
    port_sec = np.array(port)
    rw_sec = np.array(rw[:stateNum + 1])
    rw_sec = np.array(list(reversed(rw_sec)))
    est = np.dot(port_sec, rw_sec.T)
    return est
```

### Processo do Filtro de Kalman

A função `kfilter(rw, numStates, seq)` aplica o Filtro de Kalman ao sinal ruidoso, calculando o sinal filtrado e os valores de NMSE. O NMSE é calculado pela fórmula:

```math
NMSE = 10 \cdot \log_{10} \left( \frac{\text{MSE}}{\sigma_d^2} \right)
```

onde MSE é o erro quadrático médio:

```math
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (filtered[i] - rw[i])^2
```

e $\sigma_d^2$ é a variância do sinal ruidoso.

```python
def kfilter(rw, numStates, seq):
    i = 0 
    filtered = []
    nmse_values = []
    sigma_d_squared = np.var(rw)
    while i < numStates:
        port = portion(seq, i)
        est = estimate(port, rw, i)
        filtered.append(est)
        mse = np.mean((np.array(filtered) - np.array(rw[:len(filtered)]))**2)
        nmse = 10 * np.log10(mse / sigma_d_squared)
        nmse_values.append(nmse)
        i += 1

    return filtered, nmse_values
```

### Cálculo do Erro Médio Quadrático

A função `calculate_mse(filtered, rw)` calcula o Erro Médio Quadrático (MSE) do sinal filtrado em comparação com o sinal original:

```math
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (filtered[i] - rw[i])^2
```

```python
def calculate_mse(filtered, rw):
    diff_list = np.square(np.subtract(filtered, rw))
    mse = float(sum(diff_list)) / float(len(diff_list))
    print("O erro quadrático médio é ", mse)
    return mse
```

### Executando o Filtro de Kalman

A função `run_kalman_filter(original_signal, noisy_signal, numStates)` executa o Filtro de Kalman e plota os resultados.

```python
def run_kalman_filter(original_signal, noisy_signal, numStates):
    seq = fibonacci(2000)
    filtered, nmse_values = kfilter(noisy_signal, numStates, seq)
    mse = calculate_mse(filtered, original_signal)
    
    return filtered, nmse_values, mse
```

### Plotando Resultados

A função `plot(original, noisy, filtered, nmse_values)` gera gráficos para os sinais original, ruidoso e filtrado, junto com os valores de NMSE.

```python
def plot(original, noisy, filtered, nmse_values):
    x = range(1, len(filtered) + 1)
    
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x, original[:len(filtered)], color='g', label='Sinal Original')
    plt.plot(x, noisy[:len(filtered)], color='b', linestyle='--', label='Sinal Ruidoso')
    plt.plot(x, filtered, color='r', label='Sinal Filtrado')
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, nmse_values, color='g', label='NMSE')
    plt.xlabel('Tempo')
    plt.ylabel('NMSE (dB)')
    plt.legend()

    plt.tight_layout()

    plt.savefig('Exemplo Kalman 1/teste simulação 1.png')
    plt.show()
```

## Por que usar a Geração da Sequência de Fibonacci neste filtro?

A utilização da sequência de Fibonacci no cálculo dos pesos do filtro de Kalman traz algumas vantagens interessantes:

1. **Propriedades Matemáticas Únicas**: A sequência de Fibonacci possui propriedades matemáticas únicas que podem ajudar na estabilização do filtro. Cada número é a soma dos dois anteriores, o que introduz uma suavidade e continuidade natural aos pesos calculados.

2. **Distribuição Logarítmica**: Os números de Fibonacci crescem exponencialmente com uma base próxima à razão áurea (aproximadamente 1.618). Isso significa que os pesos distribuídos ao longo da sequência têm uma diminuição logarítmica, o que pode ser benéfico para a ponderação de estados em um filtro de Kalman, reduzindo gradualmente a influência de estados mais antigos.

3. **Simplicidade de Implementação**: Gerar a sequência de Fibonacci é computacionalmente simples e eficiente. Isso permite uma implementação rápida e direta dos cálculos de pesos, sem a necessidade de funções complexas ou computações intensivas.

4. **Aplicabilidade Geral**: A sequência de Fibonacci é uma escolha natural em muitos sistemas biológicos e físicos. Sua aplicação em filtros de Kalman pode refletir essa naturalidade, especialmente em sistemas onde o comportamento temporal dos estados se alinha com a progressão logarítmica dos números de Fibonacci.

## Exemplo

Abaixo está um exemplo do Filtro de Kalman aplicado a um sinal de teste:

```math
\text{sinal}(n) = \sqrt{n} + \sin\left(\frac{2 \pi \cdot 500 \cdot n}{8000}\right)
```

![Resultado do Filtro de Kalman](/KF%20com%20Peso%20Fibonacci/teste%20simulação%201.png)

Para executar o Filtro de Kalman:

```python
original_signal = np.sin(np.linspace(0, 10, 1000))
noisy_signal = original_signal + np.random.normal(0, 0.1, 1000)
filtered, nmse_values, mse = run_kalman_filter(original_signal, noisy_signal, 100)
plot(original_signal, noisy_signal, filtered, nmse_values)
```

Isso gerará um gráfico mostrando os sinais original, ruidoso e filtrado, juntamente com os valores de NMSE.

---

Sinta-se à vontade para personalizar o código e os parâmetros para se adequar aos seus requisitos e casos de uso específicos.