Vamos ajustar a formatação para garantir que o LaTeX seja corretamente renderizado. Aqui está o texto revisado:

# Geração de Sinal e Ruído

Este projeto é dedicado à geração de um sinal e ruído gaussiano branco separadamente e sua posterior combinação. A implementação é feita em Python, utilizando as bibliotecas `numpy` e `matplotlib`.

## Descrição

O principal objetivo é criar um sinal limpo, gerar ruído gaussiano branco com uma relação sinal-ruído (SNR) específica, e então combinar ambos. O resultado final é visualizado através de gráficos.

## Detalhamento das Fórmulas

### Sinal

O sinal é gerado como a soma de uma função raiz quadrada e uma senoide. A fórmula utilizada é:

$$
\text{sinal}(n) = \sqrt{n} + \sin\left(\frac{2 \pi \cdot 500 \cdot n}{8000}\right)
$$

onde:
- \( n \) representa o vetor de tempo.

### Ruído Gaussiano Branco

Para gerar o ruído gaussiano branco com a potência adequada, seguimos os seguintes passos:

1. **Calcular a potência do sinal**:

$$
P_{\text{sinal}} = \frac{1}{N} \sum_{i=1}^{N} (\text{sinal}(n_i))^2
$$

2. **Converter a SNR de decibéis para uma escala linear**:

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

### Combinação do Sinal com o Ruído

O sinal combinado com o ruído é obtido pela simples soma do sinal com o ruído gerado:

$$
\text{sinal\_com\_ruido} = \text{sinal} + \text{ruído}
$$