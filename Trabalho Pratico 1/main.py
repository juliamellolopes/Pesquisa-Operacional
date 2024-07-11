import sys
import numpy as np

def ler_arquivo_entrada(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
        
    n, m = map(int, linhas[0].split())
    c = np.array(list(map(float, linhas[1].split())))
    A = np.array([list(map(float, linha.split())) for linha in linhas[2:m+2]])
    b = np.array(list(map(float, linhas[m+2].split())))
    
    return n, m, c, A, b

def inicializar_simplex(A, b, c):
    m, n = A.shape
    tabela = np.zeros((m + 1, n + m + 1))
    tabela[:m, :n] = A
    tabela[:m, n:n + m] = np.eye(m)
    tabela[:m, -1] = b
    tabela[-1, :n] = -c
    return tabela

def simplex(n, m, c, A, b):
    tabela = inicializar_simplex(A, b, c)
    print(tabela)

def main(caminho_arquivo):
    n, m, c, A, b = ler_arquivo_entrada(caminho_arquivo)
    simplex(n, m, c, A, b)

if __name__ == '__main__':
    main(sys.argv[1])
