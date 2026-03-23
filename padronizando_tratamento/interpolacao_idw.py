import os 
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from juntando_dataframe import df
from scipy.stats import zscore
from tqdm import tqdm
import time 
from pykrige.ok import OrdinaryKriging 

df_original = df.copy()
start = time.time()
tqdm.pandas()

#VARIAVEL = "Umidade Relativa do Ar a 2m"


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def krigagem(subset, lat_ref, lon_ref, VARIAVEL):
    subset = subset.dropna(subset=[VARIAVEL])

    # krigagem precisa de pelo menos 3 pontos
    if len(subset) < 3:
        return np.nan

    lons = subset["longitude"].values
    lats = subset["latitude"].values
    valores = subset[VARIAVEL].values

    # se todos os valores forem iguais, retorna direto
    if np.all(valores == valores[0]):
        return valores[0]

    try:
        ok = OrdinaryKriging(
            lons,
            lats,
            valores,
            variogram_model="spherical",  # pode testar "gaussian" ou "linear"
            verbose=False,
            enable_plotting=False,
        )
        z, _ = ok.execute("points", [lon_ref], [lat_ref])
        return float(z[0])
    except Exception:
        # se a krigagem falhar por qualquer motivo, retorna nan
        return np.nan

def idw(subset, lat_ref, lon_ref,VARIAVEL, p=2):
    
    # remover NaN
    subset = subset.dropna(subset=[VARIAVEL])
    
    if len(subset) == 0:
        return np.nan
    
    distancias = haversine(
        lat_ref,
        lon_ref,
        subset["latitude"].values,
        subset["longitude"].values
    )
    
    # evitar divisão por zero
    distancias[distancias == 0] = 0.0001
    
    pesos = 1 / (distancias ** p)
    
    return np.sum(pesos * subset[VARIAVEL]) / np.sum(pesos)



def preencher(row, df,VARIAVEL):
    
    if not pd.isna(row[VARIAVEL]):
        return row[VARIAVEL]
    
    ano = row["ano"]
    mes = row["mes"]
    dia = row["dia"]
    hora = row["hora"]
    lat = row["latitude"]
    lon = row["longitude"]
    
    # 🔹 Nível 1
    subset = df[
        (df["ano"] == ano) &
        (df["mes"] == mes) &
        (df["dia"] == dia) &
        (df["hora"] == hora) &
        (df["id"] != row["id"])
    ]
    
    valor = idw(subset, lat, lon,VARIAVEL)
    if not np.isnan(valor):
        return valor
    
    
    # 🔹 Nível 2 (mesmo dia, horas próximas ±1)

    dif = np.abs(df["hora"] - hora)
    dif_circular = np.minimum(dif, 24 - dif)
    subset = df[
        (df["ano"] == ano) &
        (df["mes"] == mes) &
        (df["dia"] == dia) &
        (dif_circular.between(1,1)) &
        (df["id"] != row["id"])
    ]
    
    valor = idw(subset, lat, lon,VARIAVEL)
    if not np.isnan(valor):
        return valor
    
    
    # 🔹 Nível 3 (mesmo mes, mesma hora)
    subset = df[
        (df["ano"] == ano) &
        (df["mes"] == mes) &
        (df["dia"] != dia) &        # ← dia diferente
        (dif_circular.between(1, 1)) &  # ← horas próximas ±1
        (df["id"] != row["id"])
    ]
    
    valor = idw(subset, lat, lon,VARIAVEL)
    if not np.isnan(valor):
        return valor
    
    return np.nan

colunas = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    
]

nulos_antes = df[colunas].isnull().sum()

for col in colunas:
    tqdm.pandas(desc=col) 
    df[col] = df.progress_apply(
        lambda row, c=col: preencher(row, df_original, c),
        axis=1,
    )

nulos_depois = df[colunas].isnull().sum()

resultado = pd.DataFrame({
    "Antes": nulos_antes,
    "Depois": nulos_depois,
    "Preenchidos": nulos_antes - nulos_depois
})

print(resultado)
end = time.time()
print(f"tempo {end-start} segundos")

df.to_csv('arquivo_interpolado_idw.csv',index=False)


