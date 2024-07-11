import numpy as np
import sys
import time

def ler_arquivo(caminho):
    with open(caminho, 'r') as f:
        linhas = f.readlines()

    # Lê a primeira linha com os valores de n e m
    n, m = map(int, linhas[0].strip().split())

    # Lê a linha com os coeficientes da função objetivo
    c = np.array(list(map(float, linhas[1].strip().split())))

    # Lê as próximas m linhas com os coeficientes das restrições
    A = []
    b = []
    for i in range(2, 2 + m):
        linha = list(map(float, linhas[i].strip().split()))
        A.append(linha[:-1])
        b.append(linha[-1])
    A = np.array(A)
    b = np.array(b)

    return n, m, c, A, b

def inicializar_simplex(A, b, c):
    m, n = A.shape

    # Cria a tabela simplex inicial
    tabela = np.zeros((m + 1, n + m + 1))
    tabela[:m, :n] = A
    tabela[:m, n:n + m] = np.eye(m)
    tabela[:m, -1] = b
    tabela[-1, :n] = -c

    return tabela

def encontrar_coluna_pivot(tabela):
    return np.argmin(tabela[-1, :-1])

def encontrar_linha_pivot(tabela, coluna_pivot):
    m = tabela.shape[0] - 1
    razões = tabela[:m, -1] / tabela[:m, coluna_pivot]
    razões[razões <= 0] = np.inf
    return np.argmin(razões)

def executar_simplex(tabela):
    m, n = tabela.shape
    n -= 1

    iteração = 0
    while np.any(tabela[-1, :-1] < 0):
        iteração += 1

        coluna_pivot = encontrar_coluna_pivot(tabela)
        linha_pivot = encontrar_linha_pivot(tabela, coluna_pivot)

        if np.all(tabela[:m - 1, coluna_pivot] <= 0):
            raise ValueError("Problema não tem solução finita.")

        pivot = tabela[linha_pivot, coluna_pivot]
        tabela[linha_pivot, :] /= pivot

        for i in range(m):
            if i != linha_pivot:
                tabela[i, :] -= tabela[i, coluna_pivot] * tabela[linha_pivot, :]

        yield iteração, tabela

def simplex(n, m, c, A, b):
    tabela = inicializar_simplex(A, b, c)

    start_time = time.time()
    for iteração, tabela in executar_simplex(tabela):
        tempo_decorrido = time.time() - start_time
        valor_objetivo = -tabela[-1, -1]
        print(f"Iteração: {iteração}")
        print(f"Tempo(s): {tempo_decorrido:.4f}")
        print(f"Objetivo: {valor_objetivo:.4f}\n")

    tempo_total = time.time() - start_time
    valor_objetivo = -tabela[-1, -1]
    variáveis_decisão = tabela[:-1, -1]
    print(f"Solução ótima encontrada em {tempo_total:.4f} segundos!")
    print(f"Função objetivo é {valor_objetivo:.4f}.")
    for i in range(n):
        print(f"x[{i + 1}] = {variáveis_decisão[i]:.4f}")

def main(caminho):
    n, m, c, A, b = ler_arquivo(caminho)
    simplex(n, m, c, A, b)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python programa.py <caminho_do_arquivo>")
    else:
        main(sys.argv[1])
