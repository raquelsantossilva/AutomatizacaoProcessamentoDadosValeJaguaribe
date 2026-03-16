
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


_features_env = os.environ.get("FEATURES")
_alvo_env     = os.environ.get("COLUNA_ALVO")

if _features_env:
    FEATURES = json.loads(_features_env)
    logger.info("FEATURES recebidas da DAG: %s", FEATURES)
else:
    FEATURES = [
        "Umidade Relativa do Ar Mínima a 2m",
        "Velocidade Máxima do Vento 10m",
        "Fluxo de Calor no Solo",
    ]
    logger.warning("[AVISO] Env var FEATURES não encontrada. Usando fallback: %s", FEATURES)

if _alvo_env:
    COLUNA_ALVO = _alvo_env
    logger.info("COLUNA_ALVO recebida da DAG: %s", COLUNA_ALVO)
else:
    COLUNA_ALVO = "Temperatura do Ar a 2m"
    logger.warning("[AVISO] Env var COLUNA_ALVO não encontrada. Usando fallback: %s", COLUNA_ALVO)

COLUNAS_SVM = list(dict.fromkeys([COLUNA_ALVO] + FEATURES))


caminho_pasta  = os.environ.get("DATA_DIR",      "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Normalized")
pasta_saida    = Path(os.environ.get("PROCESSED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/processed"))
execution_date = os.environ.get("EXECUTION_DATE", "no-date")
pasta_saida.mkdir(parents=True, exist_ok=True)


arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .csv encontrado em: {caminho_pasta}")

df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

logger.info("Dados carregados: %d linhas", len(df))


colunas_presentes = [c for c in COLUNAS_SVM if c in df.columns]
colunas_ausentes  = [c for c in COLUNAS_SVM if c not in df.columns]

if colunas_ausentes:
    logger.warning("[AVISO] Colunas não encontradas (serão ignoradas): %s", colunas_ausentes)

if not colunas_presentes:
    raise ValueError("Nenhuma coluna de feature encontrada no dataset.")

logger.info("Colunas usadas no SVM: %s", colunas_presentes)

X_df = df[colunas_presentes].fillna(df[colunas_presentes].median())

X      = X_df


logger.info("Treinando One-Class SVM (nu=0.05, kernel=rbf)...")

svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
pred = svm.fit_predict(X)

outlier_mask = pred == -1

df["svm_outlier"] = outlier_mask.astype(int)

n_total    = len(df)
n_outliers = int(outlier_mask.sum())

logger.info("Total de registros : %d", n_total)
logger.info("Outliers detectados: %d (%.1f%%)", n_outliers, 100 * n_outliers / n_total)


saida = pasta_saida / f"svm_{execution_date}.csv"
df.to_csv(saida, index=False)
logger.info("Arquivo salvo: %s", saida)