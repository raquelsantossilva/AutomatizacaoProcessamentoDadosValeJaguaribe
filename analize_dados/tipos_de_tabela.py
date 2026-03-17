import os
import pandas as pd

caminho = r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\dados_funceme_jaguaribe"

tipos_tabelas = {}

for arquivo in os.listdir(caminho):
    if arquivo.endswith(".csv"):
        caminho_completo = os.path.join(caminho, arquivo)
        
        df = pd.read_csv(caminho_completo, nrows=0)
        colunas = tuple(df.columns)
        
        if colunas not in tipos_tabelas:
            tipos_tabelas[colunas] = []
        
        tipos_tabelas[colunas].append(arquivo)


print("Total de tipos diferentes:", len(tipos_tabelas))

for i, (colunas, arquivos) in enumerate(tipos_tabelas.items(), 1):
    print(f"\n=== Tipo {i} ===")
    print(f"Quantidade de arquivos: {len(arquivos)}")
    
    print("Arquivos:")
    for nome in arquivos:
        print("  -", nome)
    
    print("\nCabeçalho:")
    for coluna in colunas:
        print("  -", coluna)
