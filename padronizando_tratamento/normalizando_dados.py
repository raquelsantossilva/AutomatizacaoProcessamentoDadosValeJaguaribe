import pandas as pd
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\padronizando_tratamento\arquivo_interpolado.csv")

colunas_para_normalizar = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    
]

scaler = RobustScaler()
df_normalizado = df.copy()
df_normalizado[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])
df_normalizado.to_csv("arquivo_normalizado.csv",index=False)
