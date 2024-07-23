import numpy as np
import time

# Definir a função f(x)
def f(x):
    return np.sin(x) + np.cos(x)

# Definir o intervalo e o número de pontos
x_values = np.linspace(0, 2 * np.pi, 1000000)

# Usando numpy.vectorize
start_time = time.time()
f_vectorized = np.vectorize(f)
y_values_vectorize = f_vectorized(x_values)
time_vectorize = time.time() - start_time

# Usando compreensão de lista
start_time = time.time()
y_values_list_comp = np.array([f(x) for x in x_values])
time_list_comp = time.time() - start_time

# Usando numpy diretamente
start_time = time.time()
y_values_numpy = f(x_values)
time_numpy = time.time() - start_time

print(f"Tempo com numpy.vectorize: {time_vectorize:.6f} segundos")
print(f"Tempo com compreensão de lista: {time_list_comp:.6f} segundos")
print(f"Tempo com numpy diretamente: {time_numpy:.6f} segundos")
