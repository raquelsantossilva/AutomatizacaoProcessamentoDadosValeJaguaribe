from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Leitura das variáveis de ambiente
    # ------------------------------------------------------------------ #
    data_dir       = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"))
    processed_dir  = Path(os.environ.get("PROCESSED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/processed"))
    execution_date = os.environ.get("EXECUTION_DATE", "no-date")
    threshold      = float(os.environ.get("ZSCORE_THRESHOLD", "3.0"))

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
    logger.info("Threshold Z-Score  : %.1f", threshold)

    # ------------------------------------------------------------------ #
    # 3. Z-Score
    # ------------------------------------------------------------------ #
    logger.info("Executando Z-Score...")
    start = time.time()

    z_scores     = np.abs(stats.zscore(df[num_cols].fillna(df[num_cols].median()), nan_policy="omit"))
    outlier_mask = (z_scores > threshold).any(axis=1)

    elapsed = round(time.time() - start, 4)

    df["zscore_outlier"] = outlier_mask.astype(int)
    n_outliers = int(df["zscore_outlier"].sum())
    n_total    = len(df)

    logger.info("Outliers detectados: %d / %d (%.2f%%)", n_outliers, n_total,
                100 * n_outliers / n_total)
    logger.info("Tempo de execução  : %.4f s", elapsed)

    # ------------------------------------------------------------------ #
    # 4. Remove outliers e salva dataset limpo em PROCESSED_DIR
    # ------------------------------------------------------------------ #
    df_limpo    = df[df["zscore_outlier"] == 0].drop(columns=["zscore_outlier"]).reset_index(drop=True)
    n_removidos = n_total - len(df_limpo)
    logger.info("Registros removidos: %d | Restantes: %d", n_removidos, len(df_limpo))

    result_file = processed_dir / f"zscore_{execution_date}.csv"
    df_limpo.to_csv(result_file, index=False)
    logger.info("Resultado salvo: %s", result_file)

    # ------------------------------------------------------------------ #
    # 5. Métricas em JSON
    # ------------------------------------------------------------------ #
    metrics = {
        "technique":    "Z-Score",
        "n_total":      n_total,
        "n_outliers":   n_outliers,
        "n_removidos":  n_removidos,
        "pct_outliers": round(100 * n_outliers / n_total, 2),
        "elapsed_s":    elapsed,
        "result_file":  str(result_file),
        "threshold":    threshold,
    }

    metrics_file = processed_dir / f"zscore_{execution_date}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info("Métricas salvas: %s", metrics_file)


if __name__ == "__main__":
    main()