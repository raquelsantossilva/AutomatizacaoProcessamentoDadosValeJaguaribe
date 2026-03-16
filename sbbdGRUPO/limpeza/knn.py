import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ── Variáveis recebidas da DAG via env var ────────────────────────────────────

_features_env  = os.environ.get("FEATURES")
_alvo_env      = os.environ.get("COLUNA_ALVO")

if _features_env:
    FEATURES = json.loads(_features_env)
    print(f"FEATURES recebidas da DAG: {FEATURES}")
else:
    FEATURES = [
        "Umidade Relativa do Ar Mínima a 2m",
        "Velocidade Máxima do Vento 10m",
        "Fluxo de Calor no Solo",
    ]
    print(f"[AVISO] Env var FEATURES não encontrada. Usando fallback: {FEATURES}")

if _alvo_env:
    COLUNA_ALVO = _alvo_env
    print(f"COLUNA_ALVO recebida da DAG: {COLUNA_ALVO}")
else:
    COLUNA_ALVO = "Temperatura do Ar a 2m"
    print(f"[AVISO] Env var COLUNA_ALVO não encontrada. Usando fallback: {COLUNA_ALVO}")

COLUNAS_KNN = list(dict.fromkeys([COLUNA_ALVO] + FEATURES))


caminho_pasta  = os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Normalized")
pasta_saida    = Path(os.environ.get("PROCESSED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Processed"))
execution_date = os.environ.get("EXECUTION_DATE", "no-date")
pasta_saida.mkdir(parents=True, exist_ok=True)


arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .csv encontrado em: {caminho_pasta}")

df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

print(f"Dados carregados: {len(df):,} linhas")

# ── Seleciona colunas para o KNN ──────────────────────────────────────────────

colunas_presentes = [c for c in COLUNAS_KNN if c in df.columns]
colunas_ausentes  = [c for c in COLUNAS_KNN if c not in df.columns]

if colunas_ausentes:
    print(f"[AVISO] Colunas não encontradas (serão ignoradas): {colunas_ausentes}")

if not colunas_presentes:
    raise ValueError("Nenhuma coluna de feature encontrada no dataset.")

print(f"Colunas usadas no KNN: {colunas_presentes}")

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

# ── Visualização — uma variável por subplot ───────────────────────────────────

n_cols = len(colunas_presentes)
fig, axes = plt.subplots(n_cols, 1, figsize=(16, n_cols * 3), sharex=True)

if n_cols == 1:
    axes = [axes]

for ax, col in zip(axes, colunas_presentes):
    ax.scatter(df.index[~outlier_mask], df.loc[~outlier_mask, col],
               s=1, alpha=0.2, color="steelblue")
    ax.scatter(df.index[outlier_mask], df.loc[outlier_mask, col],
               s=2, alpha=0.7, color="red")
    ax.set_ylabel(col, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4)

handles = [
    plt.scatter([], [], s=10, color="steelblue", alpha=0.6, label="Normal"),
    plt.scatter([], [], s=10, color="red",       alpha=0.8, label="Outlier KNN"),
]
fig.legend(handles=handles, loc="lower right", fontsize=9, markerscale=2)
fig.suptitle("KNN — Outliers por variável (contaminação 5%)", fontsize=13)
plt.tight_layout()

# ── Salva CSV na pasta correta ────────────────────────────────────────────────

saida = pasta_saida / f"knn_{execution_date}.csv"
df.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")