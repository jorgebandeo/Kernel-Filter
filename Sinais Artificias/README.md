# Geração de Sinal e Ruído

Este projeto é dedicado à geração de um sinal e ruído gaussiano branco separadamente e sua posterior combinação. A implementação é feita em Python, utilizando as bibliotecas `numpy` e `matplotlib`.

## Descrição

O principal objetivo é criar um sinal limpo, gerar ruído gaussiano branco com uma relação sinal-ruído (SNR) específica, e então combinar ambos. O resultado final é visualizado através de gráficos.

## Detalhamento das Fórmulas

### Sinal

O sinal é gerado como a soma de uma função raiz quadrada e uma senoide. A fórmula utilizada é:

```math
\text{sinal}(n) = \sqrt{n} + \sin\left(\frac{2 \pi \cdot 500 \cdot n}{8000}\right)
```

onde:
- \( n \) representa o vetor de tempo.

### Ruído Gaussiano Branco

Para gerar o ruído gaussiano branco com a potência adequada, seguimos os seguintes passos:

1. **Calcular a potência do sinal**:

```math
P_{\text{sinal, linear}} = \frac{1}{N} \sum_{i=1}^{N} (\text{sinal}(n_i))^2
```

2. **Converter o SNR de dB para linear**:

```math
SNR_{\text{dB}} = 10 \text{log}_{10} (SNR_{\text{linear}})
```

Dividimos ambos lados por 10:
```math
\frac{SNR_{\text{dB}}}{10} = \text{log}_{10} (SNR_{\text{linear}})
```
Aplicar a operação inversa do logaritmo (exponenciação) para ambos os lados:
```math
10^{\frac{SNR_{\text{dB}}}{10}} = 10^{\text{log}_{10} (SNR_{\text{linear}})}
```

Pela regra:

```math
x = 10^{\text{log}_{10} (x)}
```
```math
SNR_{\text{linear}} = 10^{\frac{SNR_{\text{dB}}}{10}} 
```

3. **Calcular a potência em dB do ruído**:

```math
SNR_{\text{linear}} = \frac{P_{\text{sina, linear}}}{P_{\text{ruído, linear}}}
```

```math
P_{\text{ruído, linear}} = \frac{P_{\text{sina, linear}}}{SNR_{\text{linear}}}
```

4. **Gerar o ruído gaussiano branco**:

```math
\text{ruído} \sim \mathcal{N}(0, \sqrt{P_{\text{ruído}}})
```

### Combinação do Sinal com o Ruído

O sinal combinado com o ruído é obtido pela simples soma do sinal com o ruído gerado:

```math
\text{sinal\_com\_ruido} = \text{sinal} + \text{ruído}
```

