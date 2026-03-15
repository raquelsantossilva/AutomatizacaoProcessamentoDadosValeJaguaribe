from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import pandas as pd

def normalizar_dados(dados):
    scaler = StandardScaler()
    colunas_para_normalizar = [
        "Temperatura do Ar a 2m",
        "Umidade Relativa do Ar a 2m",
        "Velocidade Máxima do Vento 10m",
        "Direção do Vento 10m",
        "Fluxo de Calor no Solo"
    ]
    
    dados_normalizados = dados.copy()
    dados_normalizados[colunas_para_normalizar] = scaler.fit_transform(dados[colunas_para_normalizar])
    
    return dados_normalizados

dados = pd.read_csv("/home/raquel/programacao/estudos/sbbdGRUPO/airflow/interpolated/interpolacao_todas_variaveis.csv")
csv_normalizados = normalizar_dados(dados)
interpolated_dir = Path(os.environ.get("NORMALIZED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/normalized"))
interpolated_dir.mkdir(parents=True, exist_ok=True)
csv_normalizados.to_csv(interpolated_dir / "dados_normalizados.csv", index=False)