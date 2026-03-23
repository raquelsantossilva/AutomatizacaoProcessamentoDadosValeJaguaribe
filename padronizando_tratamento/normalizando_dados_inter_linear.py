import pandas as pd
from sklearn.preprocessing import RobustScaler
from juntando_dataframe import df

colunas_para_normalizar = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    
]

nulos_antes = df[colunas_para_normalizar].isnull().sum()

df = df.sort_values("data")  # garante ordem temporal

df[colunas_para_normalizar] = (
    df.groupby("id")[colunas_para_normalizar]
    .transform(lambda x: x.interpolate(method="linear", limit=3))
)

nulos_depois = df[colunas_para_normalizar].isnull().sum()
print("Nulos preenchidos:")
print(nulos_antes - nulos_depois)
print(f"\nNulos restantes:\n{nulos_depois}")

scaler = RobustScaler()
df_normalizado = df.copy()
df_normalizado[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])
df_normalizado.to_csv("arquivo_normalizado_linear.csv",index=False)
