import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

caminho_pasta = "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Normalized"

arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

num_cols = df.select_dtypes(include="number").columns.tolist()

# Preenche NaN com a mediana para o KNN não quebrar
X = df[num_cols].fillna(df[num_cols].median())

# ── KNN ──────────────────────────────────────────────────────────────────────
# Calcula a distância média aos k vizinhos mais próximos para cada ponto.
# Pontos com distância alta são candidatos a outlier.

k = 5
knn = NearestNeighbors(n_neighbors=k + 1)  # +1 porque o próprio ponto é incluído
knn.fit(X)

distancias, _ = knn.kneighbors(X)
# Ignora a distância 0 (o próprio ponto) e calcula a média dos k vizinhos
score_knn = distancias[:, 1:].mean(axis=1)

# Define o threshold pela contaminação de 5%
threshold = np.percentile(score_knn, 100 * (1 - 0.05))
outlier_mask = score_knn > threshold

df["knn_score"]   = score_knn
df["knn_outlier"] = outlier_mask.astype(int)

n_total    = len(df)
n_outliers = int(outlier_mask.sum())

print(f"Total de registros : {n_total:,}")
print(f"Outliers detectados: {n_outliers:,} ({100 * n_outliers / n_total:.1f}%)")
print(f"Threshold KNN score: {threshold:.4f}")

# ── Visualização ──────────────────────────────────────────────────────────────

grupos = {
    "Temperatura do Ar": [
        "Temperatura do Ar a 2m", "Temperatura Máxima do Ar a 2m",
        "Temperatura Mínima do Ar a 2m"
    ],
    "Umidade do Ar": [
        "Umidade Relativa do Ar a 2m", "Umidade Relativa do Ar Máxima a 2m",
        "Umidade Relativa do Ar Mínima a 2m"
    ],
    "Pressão Atmosférica": [
        "Pressão Atmosférica", "Pressão Atmosférica Máxima",
        "Pressão Atmosférica Mínima"
    ],
    "Vento 10m": [
        "Velocidade Máxima do Vento 10m", "Velocidade Média do Vento 10m",
        "Direção do Vento 10m"
    ],
    "Temperatura do Solo": [
        "Temperatura do Solo 5cm", "Temperatura do Solo 10cm",
        "Temperatura do Solo 30cm"
    ],
    "Fluxo de Calor e Umidade do Solo": [
        "Fluxo de Calor no Solo", "Umidade do Solo 5cm"
    ],
    "Precipitação": [
        "Precipitação Pluviométrica"
    ],
    "Radiação": [
        "Radiação", "Radiação Incidente Total"
    ],
}

cores = ["steelblue", "darkorange", "green", "purple",
         "brown", "teal", "olive", "crimson"]

n_grupos = len(grupos)
fig, axes = plt.subplots(n_grupos, 1, figsize=(16, n_grupos * 4), sharex=True)

for ax, (titulo, colunas), cor in zip(axes, grupos.items(), cores):
    colunas_presentes = [c for c in colunas if c in df.columns]

    for col in colunas_presentes:
        ax.scatter(df.index[~outlier_mask], df.loc[~outlier_mask, col],
                   s=1, alpha=0.2, color=cor)
        ax.scatter(df.index[outlier_mask], df.loc[outlier_mask, col],
                   s=2, alpha=0.7, color="red")

    ax.set_title(titulo, fontsize=9, fontweight="bold", pad=3)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4)
    ax.annotate(", ".join(colunas_presentes), xy=(0.01, 0.97),
                xycoords="axes fraction", fontsize=6, va="top", color="gray")

handles = [
    plt.scatter([], [], s=10, color="steelblue", alpha=0.6, label="Normal"),
    plt.scatter([], [], s=10, color="red",       alpha=0.8, label="Outlier KNN"),
]
fig.legend(handles=handles, loc="lower right", fontsize=9, markerscale=2)
fig.suptitle("KNN — Outliers por grupo de variáveis (contaminação 5%)", fontsize=13)
plt.tight_layout()
#plt.show()

# ── Salva CSV com flag ────────────────────────────────────────────────────────
saida = "dados_knn_outliers.csv"
df.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")