import os 
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time() 
pasta = r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\dados_funceme_jaguaribe"


metadata = pd.read_csv(os.path.join(pasta,"stations_metadata.csv"))

lista_dfs = []

for f in os.listdir(pasta):
    if f.endswith("_data.csv"):
        print(f"lendo {f}...")
        caminho = os.path.join(pasta,f)
        df_temp = pd.read_csv(caminho)
        id_estacao = re.search(r'\d+', f).group()
        df_temp["id"] = int(id_estacao)
        lista_dfs.append(df_temp)

df = pd.concat(lista_dfs, ignore_index=True)

df = df.merge(metadata , on="id", how = "left")

df["data"] = pd.to_datetime(df["data"])

df["ano"] = df["data"].dt.year
df["mes"] = df["data"].dt.month
df["dia"] = df["data"].dt.day
df["hora"] = df["data"].dt.hour

end = time.time()
print(f"tempo : {end-start :.4f} segundos")

print(len(df))
print("dataframe pronto")

colunas = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    
]

print(len(df))

