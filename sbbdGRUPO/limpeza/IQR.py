from __future__ import annotations
from datetime import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Leitura das variáveis de ambiente
    # ------------------------------------------------------------------ #
    data_dir       = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/normalized"))
    processed_dir  = Path(os.environ.get("PROCESSED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/processed"))
    execution_date = os.environ.get("EXECUTION_DATE", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    fator          = float(os.environ.get("IQR_FACTOR", "1.5"))

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
    logger.info("Fator IQR: %.1f", fator)

    # ------------------------------------------------------------------ #
    # 3. IQR — marca outlier se qualquer coluna numérica estiver fora
    # ------------------------------------------------------------------ #
    logger.info("Executando IQR...")
    start = time.time()

    outlier_mask = pd.Series(False, index=df.index)

    for col in num_cols:
        serie = df[col].dropna()
        Q1    = serie.quantile(0.25)
        Q3    = serie.quantile(0.75)
        iqr   = Q3 - Q1
        lim_inf = Q1 - fator * iqr
        lim_sup = Q3 + fator * iqr
        fora    = (df[col] < lim_inf) | (df[col] > lim_sup)
        outlier_mask = outlier_mask | fora
        logger.info("  %s: limites [%.3f, %.3f] | outliers: %d",
                    col, lim_inf, lim_sup, int(fora.sum()))

    elapsed = round(time.time() - start, 4)

    df["iqr_outlier"] = outlier_mask.astype(int)
    n_outliers = int(df["iqr_outlier"].sum())
    n_total    = len(df)

    logger.info("Outliers detectados: %d / %d (%.2f%%)", n_outliers, n_total,
                100 * n_outliers / n_total)
    logger.info("Tempo de execução  : %.4f s", elapsed)

    # ------------------------------------------------------------------ #
    # 4. Remove outliers e salva dataset limpo em PROCESSED_DIR
    # ------------------------------------------------------------------ #
    df_limpo    = df[df["iqr_outlier"] == 0].drop(columns=["iqr_outlier"]).reset_index(drop=True)
    n_removidos = n_total - len(df_limpo)
    logger.info("Registros removidos: %d | Restantes: %d", n_removidos, len(df_limpo))

    result_file = processed_dir / f"IQR_{execution_date}.csv"
    df_limpo.to_csv(result_file, index=False)
    logger.info("Resultado salvo: %s", result_file)

    # ------------------------------------------------------------------ #
    # 5. Métricas em JSON
    # ------------------------------------------------------------------ #
    metrics = {
        "technique":    "IQR",
        "n_total":      n_total,
        "n_outliers":   n_outliers,
        "n_removidos":  n_removidos,
        "pct_outliers": round(100 * n_outliers / n_total, 2),
        "elapsed_s":    elapsed,
        "result_file":  str(result_file),
        "iqr_factor":   fator,
    }

    metrics_file = processed_dir / f"IQR_{execution_date}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info("Métricas salvas: %s", metrics_file)


if __name__ == "__main__":
    main()