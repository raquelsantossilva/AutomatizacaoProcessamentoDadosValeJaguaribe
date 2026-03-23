
from __future__ import annotations
import glob
import json
import logging
import os
from pathlib import Path
import pandas as pd
from sklearn.svm import OneClassSVM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


caminho_pasta  = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\arquivo_normalizado_idw.csv")
pasta_saida    = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_inter_idw")
pasta_saida.mkdir(parents=True, exist_ok=True)



df = pd.read_csv(caminho_pasta)
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

logger.info("Dados carregados: %d linhas", len(df))


colunas_presentes =[
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    ]
logger.info("Colunas usadas no SVM: %s", colunas_presentes)

X_df = df[colunas_presentes].fillna(df[colunas_presentes].median())

X = X_df


logger.info("Treinando One-Class SVM (nu=0.05, kernel=rbf)...")

svm = OneClassSVM(kernel="rbf", nu=0.05, gamma=0.1)
pred = svm.fit_predict(X)

outlier_mask = pred == -1

df["svm_outlier"] = outlier_mask.astype(int)

n_total    = len(df)
n_outliers = int(outlier_mask.sum())

logger.info("Total de registros : %d", n_total)
logger.info("Outliers detectados: %d (%.1f%%)", n_outliers, 100 * n_outliers / n_total)

df_limpo = df[df["svm_outlier"]==0].drop(columns="svm_outlier").reset_index(drop=True)
n_removidos = n_total - len(df_limpo)
saida = pasta_saida / f"svm.csv"
df_limpo.to_csv(saida, index=False)
logger.info("Arquivo salvo: %s", saida)

#----------json--------------------------------------

metrics = {
        "technique":    "SVM",
        "n_total":      n_total,
        "n_outliers":   n_outliers,
        "n_removidos":  n_removidos,
        "pct_outliers": round(100 * n_outliers / n_total, 2),
        "result_file":  str(saida),
    }

metrics_file = pasta_saida / f"SVM_metrics.json"
metrics_file.write_text(json.dumps(metrics, indent=2))
