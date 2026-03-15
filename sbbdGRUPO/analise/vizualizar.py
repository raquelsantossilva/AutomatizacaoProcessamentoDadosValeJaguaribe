import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

caminho_pasta = "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"

arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

colunas = [c for c in df.columns if c != "data"]

Q1 = df[colunas].quantile(0.25)
Q3 = df[colunas].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

fig, axes = plt.subplots(len(colunas), 1, figsize=(14, len(colunas) * 3), sharex=True)

for ax, col in zip(axes, colunas):
    mask_outlier = (df[col] < limite_inferior[col]) | (df[col] > limite_superior[col])

    ax.scatter(df.index[~mask_outlier], df.loc[~mask_outlier, col], s=2, color="blue", alpha=0.4, label="Normal")
    ax.scatter(df.index[mask_outlier],  df.loc[mask_outlier,  col], s=2, color="red",  alpha=0.6, label="Outlier")
    ax.set_ylabel(col, fontsize=7)
    ax.grid(True)

axes[0].legend(markerscale=3)
plt.xlabel("Amostras")
plt.suptitle("IQR — Outliers por variável")
plt.tight_layout()
plt.show()