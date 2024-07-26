### README.md

# Algoritmo KFLMS para Filtragem de Sinais Ruidosos

Este repositório contém uma implementação do algoritmo Kernel Least Mean Squares (KLMS) para a filtragem de diferentes tipos de ruído em um sinal senoidal puro. O objetivo é demonstrar a eficácia do KLMS na atenuação de diversos ruídos adicionados ao sinal original.

## Tipos de Ruídos Gerados

O código gera diferentes tipos de ruídos para testar a eficácia do algoritmo KLMS. Abaixo estão os detalhes de cada tipo de ruído, juntamente com as fórmulas matemáticas e o código usado para gerá-los.

### 1. Ruído Branco
O ruído branco é um sinal com uma densidade espectral de potência constante, ou seja, possui a mesma potência em todas as frequências.

**Fórmula Matemática:**
$$\text{white\_noise}(n) = \sin(2 \pi f n) + N(0, 1)$$
onde $N(0, 1)$ representa uma distribuição normal com média 0 e variância 1.

**Código:**
```python
white_noise = sinal_puro + np.random.normal(0, 1, len(n))
```

### 2. Ruído Gaussiano
O ruído gaussiano tem uma distribuição normal com média zero e desvio padrão específico.

**Fórmula Matemática:**
$$\text{gaussian\_noise}(n) = \sin(2 \pi f n) + N(0, 0.5)$$
onde $N(0, 0.5)$ representa uma distribuição normal com média 0 e variância 0.25.

**Código:**
```python
gaussian_noise = sinal_puro + np.random.normal(0, 0.5, len(n))
```

### 3. Ruído de Banda Limitada
O ruído de banda limitada é um sinal ruidoso dentro de uma faixa de frequências específica.

**Fórmula Matemática:**
$$\text{band\_limited\_noise}(n) = \sin(2 \pi f n) + \sin(2 \pi 150 n) \cdot N(0, 0.5)$$
onde $N(0, 0.5)$ é um ruído gaussiano.

**Código:**
```python
band_limited_noise = sinal_puro + np.sin(2 * np.pi * 150 * n) * np.random.normal(0, 0.5, len(n))
```

### 4. Ruído Impulsivo
O ruído impulsivo consiste em picos de alta amplitude que ocorrem esporadicamente.

**Fórmula Matemática:**
$$\text{impulse\_noise}(n) = \sin(2 \pi f n)$$
$$\text{impulse\_noise}[\text{indices}] += N(0, 5)$$
onde $\text{indices}$ são posições aleatórias no sinal.

**Código:**
```python
impulse_noise = sinal_puro.copy()
impulse_indices = np.random.randint(0, len(n), 50)
impulse_noise[impulse_indices] += np.random.normal(0, 5, 50)
```

### 5. Ruído Rosa (Aproximado)
O ruído rosa possui uma densidade espectral de potência que decresce com o aumento da frequência.

**Fórmula Matemática:**
$$\text{pink\_noise}(n) = \sin(2 \pi f n) + \sum_{k=1}^{n} N(0, 1)$$

**Código:**
```python
pink_noise = sinal_puro + np.cumsum(np.random.normal(0, 1, len(n)))
```

### 6. Ruído de Fundo
O ruído de fundo é uma combinação de ruído gaussiano e uma componente sinusoidal.

**Fórmula Matemática:**
$$\text{background\_noise}(n) = \sin(2 \pi f n) + N(0, 0.2) + \sin(2 \pi 50 n) \cdot 0.2$$

**Código:**
```python
background_noise = sinal_puro + np.random.normal(0, 0.2, len(n)) + np.sin(2 * np.pi * 50 * n) * 0.2
```

## Visualização dos Resultados

O código aplica o algoritmo KLMS a cada tipo de ruído e plota os resultados, incluindo a evolução do NMSE (Erro Médio Normalizado) e PSNR (Proporção Sinal-Ruído de Pico) ao longo do tempo. A figura resultante é salva como 'teste de ruidos/teste de ruidos.png'.

```python
plt.tight_layout()
plt.savefig('teste de ruidos/teste de ruidos.png')
plt.show()
```

![Resultados](teste%20de%20ruidos.png)