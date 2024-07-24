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

2. **Converter a potência do sinal linear para dB**:

```math
P_{\text{sina, dB}} = 10\text{log}_{10} (P_{\text{sina, linear}} )
```

3. **Calcular a potência em dB do ruído**:

```math
SNR_{\text{dB}} = P_{\text{sina, dB}} - P_{\text{ruído, dB}}
```

```math
P_{\text{ruído, dB}} = P_{\text{sina, dB}} - SNR_{\text{dB}}
```
4. **Converter a potência do ruído em dB para linear**:

Para inverter a fórmula \( P_{\text{sina, dB}} = 10\log_{10} (P_{\text{sina, linear}}) \) e transformar de dB para linear:

4.1. Comece com a equação original:

```math
P_{\text{sina, dB}} = 10\log_{10} (P_{\text{sina, linear}})
```

4.2. Divida ambos os lados da equação por 10:

```math
\frac{P_{\text{sina, dB}}}{10} = \log_{10} (P_{\text{sina, linear}})
```

4.3. Utilize a operação inversa do logaritmo base 10, que é elevar 10 à potência de ambos os lados da equação:

```math
10^{\frac{P_{\text{sina, dB}}}{10}} = P_{\text{sina, linear}}
```

Portanto, a fórmula invertida para converter de dB para linear é:

```math
P_{\text{sina, linear}} = 10^{\frac{P_{\text{sina, dB}}}{10}}
```

5. **Gerar o ruído gaussiano branco**:

```math
\text{ruído} \sim \mathcal{N}(0, \sqrt{P_{\text{ruído}}})
```

### Combinação do Sinal com o Ruído

O sinal combinado com o ruído é obtido pela simples soma do sinal com o ruído gerado:

```math
\text{sinal\_com\_ruido} = \text{sinal} + \text{ruído}
```

