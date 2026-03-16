from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Leitura das variáveis de ambiente
    # ------------------------------------------------------------------ #
    data_dir       = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Normalized"))
    processed_dir  = Path(os.environ.get("PROCESSED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/processed"))
    execution_date = os.environ.get("EXECUTION_DATE", "no-date")

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. Carregamento dos dados
    # ------------------------------------------------------------------ #
    arquivos = glob.glob(str(data_dir / "*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em: {data_dir}")

    df = pd.concat((pd.read_csv(f) for f in arquivos), ignore_index=True)
    logger.info("Dados carregados: %d linhas de %d arquivos", len(df), len(arquivos))

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise ValueError("Nenhuma coluna numérica encontrada no dataset.")

    logger.info("Colunas usadas (%d): %s", len(num_cols), num_cols)

    logger.info("Executando Local Outlier Factor...")
    start = time.time()

    n_neighbors = min(3, len(df) - 1)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
    preds = model.fit_predict(df[num_cols].fillna(df[num_cols].median()))

    elapsed = round(time.time() - start, 4)

    df["lof_outlier"] = (preds == -1).astype(int)
    n_outliers = int(df["lof_outlier"].sum())
    n_total    = len(df)

    logger.info("Outliers detectados: %d / %d (%.2f%%)", n_outliers, n_total,
                100 * n_outliers / n_total)
    logger.info("Tempo de execução  : %.4f s", elapsed)

    # ------------------------------------------------------------------ #
    # 4. Remove outliers e salva dataset limpo em PROCESSED_DIR
    # ------------------------------------------------------------------ #
    df_limpo    = df[df["lof_outlier"] == 0].drop(columns=["lof_outlier"]).reset_index(drop=True)
    n_removidos = n_total - len(df_limpo)
    logger.info("Registros removidos: %d | Restantes: %d", n_removidos, len(df_limpo))

    result_file = processed_dir / f"lof_{execution_date}.csv"
    df_limpo.to_csv(result_file, index=False)
    logger.info("Resultado salvo: %s", result_file)

    # ------------------------------------------------------------------ #
    # 5. Métricas em JSON
    # ------------------------------------------------------------------ #
    metrics = {
        "technique":    "LOF",
        "n_total":      n_total,
        "n_outliers":   n_outliers,
        "n_removidos":  n_removidos,
        "pct_outliers": round(100 * n_outliers / n_total, 2),
        "elapsed_s":    elapsed,
        "result_file":  str(result_file),
    }

    metrics_file = processed_dir / f"lof_{execution_date}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info("Métricas salvas: %s", metrics_file)


if __name__ == "__main__":
    main()