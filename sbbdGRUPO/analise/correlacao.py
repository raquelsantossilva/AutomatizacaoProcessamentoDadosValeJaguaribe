import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diretorio = "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"

arquivos = glob.glob(f"{diretorio}/*.csv")
juntados = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos), ignore_index=True)
corr = juntados.drop(columns="data", errors='ignore').corr()

filtrada = corr[(corr > 0.5) | (corr < -0.5)]

# Remove linhas e colunas onde todos os valores são NaN
cols_relevantes = filtrada.dropna(how='all').dropna(axis=1, how='all').index
filtrada = filtrada.loc[cols_relevantes, cols_relevantes]

plt.figure(figsize=(30, 24))
sns.heatmap(
    filtrada,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)
plt.tight_layout()
plt.savefig("heatmap_correlacao.png", dpi=150)
plt.show()