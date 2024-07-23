# Comparação de Desempenho de Abordagens para Aplicar Funções em Arrays Numpy

Este projeto compara três abordagens para aplicar uma função a um array numpy e mede o desempenho de cada uma:

1. Usando `numpy.vectorize`.
2. Usando compreensão de lista.
3. Usando operações vetorizadas diretamente com `numpy`.

O arquivo **[Teste.py](Teste.py)** Contem o código usado para fazer o teste de desempenho de cada metodo.

Resultado opticos:

* Tempo com numpy.vectorize: 2.451257 segundos
* Tempo com compreensão de lista: 2.468949 segundos
* Tempo com numpy diretamente: 0.019544 segundos

Protanto o melhor caso é usar numpy diretamente para vetorizar sinais de teste a partir de funçoes.