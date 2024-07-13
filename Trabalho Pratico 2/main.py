import sys
import numpy as np
import time

def parse_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    n, m = map(int, lines[0].strip().split())
    c = np.array(list(map(float, lines[1].strip().split())))
    A = np.array([list(map(float, line.strip().split())) for line in lines[2:m+2]])
    b = A[:, -1]
    A = A[:, :-1]
    
    return n, m, c, A, b

def simplex(c, A, b):
    m, n = A.shape
    A = np.hstack([A, np.eye(m)])  # Adicionando variáveis de folga
    c = np.hstack([c, np.zeros(m)])  # Adicionando custos das variáveis de folga
    
    B = list(range(n, n + m))  # Índices das variáveis básicas
    N = list(range(n))  # Índices das variáveis não básicas
    
    iteration = 1
    
    while True:
        cb = c[B]
        cn = c[N]
        
        Ab = A[:, B]
        An = A[:, N]
        
        xb = np.linalg.solve(Ab, b)
        
        lambd = np.linalg.solve(Ab.T, cb)
        
        reduced_costs = cn - An.T @ lambd
        
        print(f"Iteração: {iteration}")
        print(f"Tempo(s): {time.time() - start_time:.4f}")
        print(f"Objetivo: {c[B] @ xb:.4f}")
        
        if all(reduced_costs >= 0):
            x = np.zeros(n + m)
            x[B] = xb
            return x[:n], c @ x
        
        entering = N[np.argmin(reduced_costs)]
        
        direction = np.linalg.solve(Ab, A[:, entering])
        
        if all(direction <= 0):
            raise ValueError("Problema ilimitado")
        
        ratios = np.array([xb[i] / direction[i] if direction[i] > 0 else np.inf for i in range(m)])
        
        leaving = B[np.argmin(ratios)]
        
        B[B.index(leaving)] = entering
        N[N.index(entering)] = leaving
        
        iteration += 1

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <caminho_para_arquivo_de_entrada>")
        return
    
    file_path = sys.argv[1]
    n, m, c, A, b = parse_input(file_path)
    
    global start_time
    start_time = time.time()
    x, obj_value = simplex(c, A, b)
    end_time = time.time()
    
    print(f"Solução ótima encontrada em {end_time - start_time:.4f} segundos!")
    print(f"Função objetivo é {obj_value:.4f}.")
    
    for i in range(len(x)):
        print(f"x[{i+1}] = {x[i]:.4f}")

if __name__ == "__main__":
    main()
