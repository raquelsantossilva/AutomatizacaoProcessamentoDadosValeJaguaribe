import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


# ── Caminhos ──────────────────────────────────────────────────────────────────

caminho_pasta  = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\arquivo_normalizado_sem_inter.csv")
pasta_saida    = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_sem_interpo")
pasta_saida.mkdir(parents=True, exist_ok=True)

# ── Carregamento ──────────────────────────────────────────────────────────────

df = pd.read_csv(caminho_pasta)
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

print(f"Dados carregados: {len(df):,} linhas")

# ── Seleciona colunas para o KNN ──────────────────────────────────────────────

colunas_presentes = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
    ]
df = df.dropna(subset=colunas_presentes).reset_index(drop=True)
X = df[colunas_presentes].fillna(df[colunas_presentes].median())

# ── KNN ───────────────────────────────────────────────────────────────────────

k = 5
knn = NearestNeighbors(n_neighbors=k + 1)
knn.fit(X)

distancias, _ = knn.kneighbors(X)
score_knn     = distancias[:, 1:].mean(axis=1)

threshold    = np.percentile(score_knn, 100 * (1 - 0.05))
outlier_mask = score_knn > threshold

df["knn_score"]   = score_knn
df["knn_outlier"] = outlier_mask.astype(int)

n_total    = len(df)
n_outliers = int(outlier_mask.sum())
print(f"\nTotal de registros : {n_total:,}")
print(f"Outliers detectados: {n_outliers:,} ({100 * n_outliers / n_total:.1f}%)")
print(f"Threshold KNN score: {threshold:.4f}")

# ── Salva CSV na pasta correta ────────────────────────────────────────────────
df_limpo = df[df["knn_outlier"]==0].drop(columns=["knn_outlier"]).reset_index(drop=True)
n_removidos = n_total - len(df_limpo)
saida = pasta_saida / f"knn_sem_inter.csv"
df_limpo.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")
#---- json -----------------------------------------------------
metrics = {
        "technique":    "KNN",
        "n_total":      n_total,
        "n_outliers":   n_outliers,
        "n_removidos":  n_removidos,
        "pct_outliers": round(100 * n_outliers / n_total, 2),
        "result_file":  str(saida),
    }

metrics_file = pasta_saida / f"KNN_metrics_sem_inter.json"
metrics_file.write_text(json.dumps(metrics, indent=2))
