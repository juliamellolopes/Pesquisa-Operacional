import sys
import numpy as np
from itertools import combinations

def ler_arquivo_entrada(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        linhas = file.readlines()
        
    n, m = map(int, linhas[0].split())
    c = np.array(list(map(float, linhas[1].split())))
    A = np.array([list(map(float, linha.split())) for linha in linhas[2:m+2]])
    b = np.array(list(map(float, linhas[m+2].split())))
    
    return n, m, c, A, b

def resolver_sistema(A, b):
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        return None

def enumerar_solucoes_basicas(n, m, c, A, b):
    indices = list(range(n))
    combinacoes = list(combinations(indices, m))
    
    solucoes = []
    
    for comb in combinacoes:
        A_basica = A[:, comb]
        x_basica = resolver_sistema(A_basica, b)
        
        if x_basica is not None:
            x = np.zeros(n)
            x[list(comb)] = x_basica
            z = np.dot(c, x)
            viavel = np.all(x >= 0)
            solucoes.append((x, z, viavel))
    
    return solucoes

def main(caminho_arquivo):
    n, m, c, A, b = ler_arquivo_entrada(caminho_arquivo)
    solucoes = enumerar_solucoes_basicas(n, m, c, A, b)
    
    solucoes_viaveis = 0
    solucoes_inviaveis = 0
    
    for x, z, viavel in solucoes:
        if viavel:
            solucoes_viaveis += 1
        else:
            solucoes_inviaveis += 1
            
        solucao_str = f"Solução: x={tuple(x)}, z={z}, {'viável' if viavel else 'inviável'}"
        if viavel and z == min([z for x, z, v in solucoes if v]):
            solucao_str += " ==> ótima"
        print(solucao_str)
    
    print(f"Soluções básicas viáveis: {solucoes_viaveis}")
    print(f"Soluções básicas inviáveis: {solucoes_inviaveis}")

if __name__ == '__main__':
    main(sys.argv[1])
