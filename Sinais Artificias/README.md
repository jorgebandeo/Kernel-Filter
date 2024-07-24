# Geração de Sinal e Ruído

Este projeto gera um sinal e ruído gaussiano branco de forma separada e, em seguida, combina ambos. O código é implementado em Python utilizando as bibliotecas `numpy` e `matplotlib`.

## Descrição

O objetivo deste projeto é gerar um sinal limpo, gerar ruído gaussiano branco com uma relação sinal-ruído (SNR) especificada, e combinar o sinal com o ruído. O resultado é visualizado através de gráficos.

## Fórmulas

### Sinal

O sinal gerado é a soma de uma raiz quadrada e uma senoide. A fórmula utilizada é:

$$
\text{sinal}(n) = \sqrt{n} + \sin\left(\frac{2 \pi \cdot 500 \cdot n}{8000}\right)
$$

onde:
- \( n \) é o vetor de tempo.

### Ruído Gaussiano Branco

Para gerar o ruído gaussiano branco com a potência correta, usamos as seguintes etapas:

1. **Calcular a potência do sinal**:

$$
P_{\text{sinal}} = \frac{1}{N} \sum_{i=1}^{N} (\text{sinal}(n_i))^2
$$

2. **Converter o SNR de decibéis para linear**:

$$
\text{SNR}_{\text{linear}} = 10^{\left(\frac{\text{SNR}_{\text{dB}}}{10}\right)}
$$

3. **Calcular a potência do ruído**:

$$
P_{\text{ruído}} = \frac{P_{\text{sinal}}}{\text{SNR}_{\text{linear}}}
$$

4. **Gerar o ruído gaussiano branco**:

$$
\text{ruído} \sim \mathcal{N}(0, \sqrt{P_{\text{ruído}}})
$$

### Sinal com Ruído

O sinal com ruído é simplesmente a soma do sinal e do ruído:

$$\text{sinal\_com\_ruido} = \text{sinal} + \text{ruido}$$