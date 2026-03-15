"""
ml_validation.py
----------------
Tarefa E — Validação do comparativo por modelo de ML supervisionado.

Estratégia:
  - Usa as predições dos 3 modelos (IF, LOF, Z-Score) + features numéricas originais.
  - O rótulo (target) é o CONSENSO: pontos marcados como outlier por TODAS as técnicas.
    Se o consenso resultar em target vazio, usa votos por MAIORIA (≥ 2 modelos).
  - Treina um Random Forest Classifier com validação cruzada 5-fold estratificada.
  - Avalia com Accuracy, Precision, Recall e F1-macro.
  - Compara cada métrica com metas configuráveis via variáveis de ambiente.
  - Sai com código != 0 (falha a tarefa no Airflow) se alguma meta não for atingida.

Variáveis de ambiente esperadas (injetadas pela DAG):
    RESULTS_DIR          : pasta com os parquets de resultado
    PREPROCESSED_FILE    : dataset pré-processado (features numéricas originais)
    EXECUTION_DATE       : data de execução (ds do Airflow)

Metas (opcionais — padrões conservadores):
    GOAL_ACCURACY        : float, padrão 0.80
    GOAL_PRECISION       : float, padrão 0.75
    GOAL_RECALL          : float, padrão 0.70
    GOAL_F1              : float, padrão 0.72
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metas de qualidade (lidas de env vars com fallback)
# ---------------------------------------------------------------------------

GOAL_ACCURACY  = float(os.environ.get("GOAL_ACCURACY",  0.80))
GOAL_PRECISION = float(os.environ.get("GOAL_PRECISION", 0.75))
GOAL_RECALL    = float(os.environ.get("GOAL_RECALL",    0.70))
GOAL_F1        = float(os.environ.get("GOAL_F1",        0.72))


def _check_goal(metric_name: str, value: float, goal: float, failures: list) -> None:
    status = "✅" if value >= goal else "❌"
    logger.info("  %s  %-12s : %.4f  (meta ≥ %.2f)", status, metric_name, value, goal)
    if value < goal:
        failures.append(f"{metric_name} = {value:.4f} abaixo da meta {goal:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir       = Path(os.environ["RESULTS_DIR"])
    preprocessed_file = Path(os.environ["PREPROCESSED_FILE"])
    execution_date    = os.environ.get("EXECUTION_DATE", "no-date")

    # ------------------------------------------------------------------ #
    # 1. Carrega predições de cada modelo
    # ------------------------------------------------------------------ #
    files = {
        "if_outlier":     results_dir / f"isolation_forest_{execution_date}.parquet",
        "lof_outlier":    results_dir / f"lof_{execution_date}.parquet",
        "zscore_outlier": results_dir / f"zscore_{execution_date}.parquet",
    }

    for col, path in files.items():
        if not path.exists():
            logger.error("Arquivo não encontrado: %s", path)
            sys.exit(1)

    pred_series = {col: pd.read_parquet(path)[[col]].squeeze()
                   for col, path in files.items()}

    predictions = pd.DataFrame(pred_series).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Features = predições + colunas numéricas originais
    # ------------------------------------------------------------------ #
    df_base  = pd.read_parquet(preprocessed_file)
    num_cols = df_base.select_dtypes(include="number").columns.tolist()

    X = pd.concat([predictions, df_base[num_cols].reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------------ #
    # 3. Define target: CONSENSO → fallback MAIORIA
    # ------------------------------------------------------------------ #
    y_consensus = predictions.all(axis=1).astype(int)
    n_outliers  = int(y_consensus.sum())
    n_total     = len(y_consensus)

    if 0 < n_outliers < n_total:
        y              = y_consensus
        target_strategy = "consenso (todos os modelos concordam)"
    else:
        logger.warning("Consenso inviável (%d outliers). Usando maioria (≥ 2 modelos).", n_outliers)
        y               = (predictions.sum(axis=1) >= 2).astype(int)
        n_outliers      = int(y.sum())
        target_strategy = "maioria (≥ 2 modelos concordam)"

    if n_outliers == 0 or n_outliers == n_total:
        logger.error("Target completamente desbalanceado — impossível treinar o modelo.")
        sys.exit(1)

    logger.info("Target: %s", target_strategy)
    logger.info("Outliers no target: %d / %d (%.2f%%)",
                n_outliers, n_total, 100 * n_outliers / n_total)

    # ------------------------------------------------------------------ #
    # 4. Validação cruzada estratificada 5-fold
    # ------------------------------------------------------------------ #
    logger.info("Treinando Random Forest com validação cruzada 5-fold...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=False,
    )

    cv_accuracy  = float(np.mean(cv_results["test_accuracy"]))
    cv_precision = float(np.mean(cv_results["test_precision_macro"]))
    cv_recall    = float(np.mean(cv_results["test_recall_macro"]))
    cv_f1        = float(np.mean(cv_results["test_f1_macro"]))
    elapsed      = round(time.time() - start, 4)

    # ------------------------------------------------------------------ #
    # 5. Treina modelo final e gera relatório detalhado
    # ------------------------------------------------------------------ #
    model.fit(X, y)
    y_pred   = model.predict(X)
    report   = classification_report(y, y_pred, target_names=["inlier", "outlier"])
    conf_mat = confusion_matrix(y, y_pred).tolist()

    feature_importance = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info("\nRelatório de classificação (modelo final):\n%s", report)
    logger.info("Top 10 features mais importantes:")
    for feat, imp in top_features:
        logger.info("  %-30s : %.4f", feat, imp)

    # ------------------------------------------------------------------ #
    # 6. Verificação de metas
    # ------------------------------------------------------------------ #
    logger.info(
        "\n"
        "════════════════════════════════════════════════\n"
        "  🎯  VERIFICAÇÃO DE METAS — CV 5-fold          \n"
        "════════════════════════════════════════════════\n"
        "  Metas configuradas via variáveis de ambiente:\n"
        "    GOAL_ACCURACY  = %.2f\n"
        "    GOAL_PRECISION = %.2f\n"
        "    GOAL_RECALL    = %.2f\n"
        "    GOAL_F1        = %.2f\n"
        "════════════════════════════════════════════════",
        GOAL_ACCURACY, GOAL_PRECISION, GOAL_RECALL, GOAL_F1,
    )

    failures: list[str] = []
    _check_goal("Accuracy",  cv_accuracy,  GOAL_ACCURACY,  failures)
    _check_goal("Precision", cv_precision, GOAL_PRECISION, failures)
    _check_goal("Recall",    cv_recall,    GOAL_RECALL,    failures)
    _check_goal("F1-macro",  cv_f1,        GOAL_F1,        failures)

    # ------------------------------------------------------------------ #
    # 7. Salva resultado completo em JSON
    # ------------------------------------------------------------------ #
    validation_result = {
        "execution_date":    execution_date,
        "model":             "RandomForestClassifier",
        "cv_folds":          5,
        "target_strategy":   target_strategy,
        "n_total":           n_total,
        "n_outliers_target": n_outliers,
        "goals": {
            "accuracy":  GOAL_ACCURACY,
            "precision": GOAL_PRECISION,
            "recall":    GOAL_RECALL,
            "f1_macro":  GOAL_F1,
        },
        "cv_metrics": {
            "accuracy":  round(cv_accuracy,  4),
            "precision": round(cv_precision, 4),
            "recall":    round(cv_recall,    4),
            "f1_macro":  round(cv_f1,        4),
        },
        "goals_achieved":     len(failures) == 0,
        "goals_failed":       failures,
        "elapsed_s":          elapsed,
        "top_10_features":    dict(top_features),
        "confusion_matrix":   conf_mat,
        "classification_report": report,
    }

    result_file = results_dir / f"ml_validation_{execution_date}.json"
    result_file.write_text(json.dumps(validation_result, indent=2, ensure_ascii=False))
    logger.info("Resultado salvo: %s", result_file)

    # ------------------------------------------------------------------ #
    # 8. Falha a tarefa se alguma meta não foi atingida
    # ------------------------------------------------------------------ #
    if failures:
        logger.error(
            "\n════════════════════════════════════════════════\n"
            "  ❌  METAS NÃO ATINGIDAS (%d/%d falharam):\n"
            "     %s\n"
            "════════════════════════════════════════════════",
            len(failures), 4, "\n     ".join(failures),
        )
        sys.exit(1)

    logger.info(
        "\n════════════════════════════════════════════════\n"
        "  ✅  TODAS AS METAS ATINGIDAS!                 \n"
        "     Accuracy  : %.4f  (≥ %.2f)\n"
        "     Precision : %.4f  (≥ %.2f)\n"
        "     Recall    : %.4f  (≥ %.2f)\n"
        "     F1-macro  : %.4f  (≥ %.2f)\n"
        "════════════════════════════════════════════════",
        cv_accuracy,  GOAL_ACCURACY,
        cv_precision, GOAL_PRECISION,
        cv_recall,    GOAL_RECALL,
        cv_f1,        GOAL_F1,
    )


if __name__ == "__main__":
    main()