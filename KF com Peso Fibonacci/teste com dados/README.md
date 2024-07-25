### README.md

# Experimentos com Filtro de Kalman usando Dados de Acelerômetro

Este repositório contém dois experimentos distintos utilizando um Filtro de Kalman com pesos baseados na sequência de Fibonacci. Os experimentos aplicam o filtro a dados reais obtidos de um acelerômetro, avaliando a eficácia da filtragem de ruído e a precisão da estimativa do sinal original.

## Descrição dos Experimentos

### Experimento 1: Filtro de Kalman Padrão

#### Modificações

1. **Dados de Entrada**: Utilizamos dados reais de um acelerômetro, extraídos da coluna `aX` de um arquivo CSV.
2. **Número de Estados**: Definido como 20.
3. **Implementação**: Aplicamos o filtro de Kalman padrão com a sequência de Fibonacci para ponderação dos estados.

#### Resultados

![Resultado do Filtro de Kalman Padrão](/KF%20com%20Peso%20Fibonacci/teste%20com%20dados/teste%20csv.png)

Neste experimento, o filtro de Kalman foi capaz de reduzir significativamente o ruído presente no sinal original do acelerômetro, resultando em uma estimativa mais suave e precisa.

### Experimento 2: Filtro de Kalman Potencializado

#### Modificações

1. **Dados de Entrada**: Utilizamos os mesmos dados reais de um acelerômetro, extraídos da coluna `aX` de um arquivo CSV.
2. **Número de Estados**: Aumentado para 200.
3. **Iterações**: O filtro foi aplicado iterativamente 20 vezes para melhorar a filtragem, embora isso tenha levado a uma redução no desempenho devido ao maior custo computacional.

#### Resultados

![Resultado do Filtro de Kalman Potencializado](/KF%20com%20Peso%20Fibonacci/teste%20com%20dados/teste%20potencializado%20csv.png)

Apesar da filtragem mais potente, a quantidade de iterações e o aumento no número de estados resultaram em uma redução no desempenho do filtro, com um custo computacional mais alto e uma menor precisão na estimativa do sinal.

## Conclusão

Ambos os experimentos demonstram a eficácia do filtro de Kalman na redução do ruído de sinais de acelerômetro. A escolha entre um filtro padrão e um potencializado depende do equilíbrio desejado entre a precisão da filtragem e o custo computacional.

