from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import TriggerRule


logger = logging.getLogger(__name__)

BASE_DIR    = Path(os.getenv("PREPROCESSING_BASE_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow"))
SCRIPTS_DIR = Path(os.getenv("SCRIPTS_DIR",            "/home/raquel/programacao/estudos/sbbdGRUPO/limpeza"))
DATA_DIR    = Path(os.getenv("DATA_DIR",               "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"))

INTERPOLATED_DIR = BASE_DIR / "interpolated"
NORMALIZED_DIR = BASE_DIR / "Normalized"
PROCESSED_DIR    = BASE_DIR / "processed"
REPORTS_DIR      = BASE_DIR / "reports"

for _dir in (INTERPOLATED_DIR, NORMALIZED_DIR, PROCESSED_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

#Script de normalizacao
NORMALIZACAO_SCRIPT =  Path(os.getenv("NORMALIZACAO_SCRIPT","/home/raquel/programacao/estudos/sbbdGRUPO/limpeza/normalizacao.py"))
# Script de interpolação
INTERPOLACAO_SCRIPT = Path(os.getenv("INTERPOLACAO_SCRIPT",
    "/home/raquel/programacao/estudos/sbbdGRUPO/limpeza/interpolacao/espacotemp.py"))

# Scripts de detecção de outliers
ISOLATION_FOREST_SCRIPT = SCRIPTS_DIR / "isolation_forest.py"
LOF_SCRIPT              = SCRIPTS_DIR / "lof.py"
ZSCORE_SCRIPT           = SCRIPTS_DIR / "zscore.py"
IQR_SCRIPT              = SCRIPTS_DIR / "IQR.py"
KNN_SCRIPT              = SCRIPTS_DIR / "knn.py"
# Coluna alvo e features
COLUNA_ALVO = "Temperatura do Ar a 2m"
FEATURES    = [
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Direção do Vento 2m",
    "Fluxo de Calor no Solo",
]



# Métodos de outlier e sufixo esperado no arquivo processado
METODOS = {
    "IQR":              "IQR",
    "zscore":           "zscore",
    "lof":              "lof",
    "isolation_forest": "isolation_forest",
}


default_args = {
    "owner":             "data-engineering",
    "depends_on_past":   False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=60),
}


def verificar_dados(**context) -> None:
    """Verifica se DATA_DIR contém CSVs antes de iniciar o pipeline."""
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise AirflowFailException(f"Nenhum arquivo .csv encontrado em: {DATA_DIR}")
    logger.info("✔ %d arquivo(s) encontrado(s) em %s", len(csv_files), DATA_DIR)
    context["ti"].xcom_push(key="data_files", value=[str(f) for f in csv_files])


def run_interpolacao(**context) -> None:
    """
    Executa o script de interpolação espaciotemporal.
    Lê de DATA_DIR e salva o CSV interpolado em INTERPOLATED_DIR.
    """
    env_vars = {
        **os.environ,
        "DATA_DIR":        str(DATA_DIR),
        "INTERPOLATED_DIR": str(INTERPOLATED_DIR),
        "EXECUTION_DATE":  context["ds"],
    }
    cmd = ["python", str(INTERPOLACAO_SCRIPT)]
    logger.info("Executando interpolação: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars)

    if result.stdout:
        logger.info("[stdout]\n%s", result.stdout)
    if result.stderr:
        logger.warning("[stderr]\n%s", result.stderr)
    if result.returncode != 0:
        raise AirflowFailException(
            f"'espacotemp.py' falhou com código {result.returncode}."
        )

    # Confirma que o CSV interpolado foi gerado
    csvs = list(INTERPOLATED_DIR.glob("*.csv"))
    if not csvs:
        raise AirflowFailException(
            f"Interpolação concluída mas nenhum CSV encontrado em {INTERPOLATED_DIR}."
        )
    logger.info("✔ Interpolação concluída — %d arquivo(s) em %s", len(csvs), INTERPOLATED_DIR)
    context["ti"].xcom_push(key="interpolated_files", value=[str(f) for f in csvs])

def run_normalizacao(**context) -> None:
    """
    Executa o script de normalização.
    Lê de INTERPOLATED_DIR e salva o CSV normalizado em NORMALIZED_DIR.
    """
    env_vars = {
        **os.environ,
        "DATA_DIR":        str(INTERPOLATED_DIR),
        "NORMALIZED_DIR":   str(NORMALIZED_DIR),
        "EXECUTION_DATE":  context["ds"],
    }
    cmd = ["python", str(NORMALIZACAO_SCRIPT)]
    logger.info("Executando normalização: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars)

    if result.stdout:
        logger.info("[stdout]\n%s", result.stdout)
    if result.stderr:
        logger.warning("[stderr]\n%s", result.stderr)
    if result.returncode != 0:
        raise AirflowFailException(
            f"'normalizacao.py' falhou com código {result.returncode}."
        )

    # Confirma que o CSV normalizado foi gerado
    csvs = list(NORMALIZED_DIR.glob("*.csv"))
    if not csvs:
        raise AirflowFailException(
            f"Normalização concluída mas nenhum CSV encontrado em {NORMALIZED_DIR}."
        )
    logger.info("✔ Normalização concluída — %d arquivo(s) em %s", len(csvs), NORMALIZED_DIR)
    context["ti"].xcom_push(key="normalized_files", value=[str(f) for f in csvs])

def _run_script(script_path: Path, context: dict) -> None:
    """Executa script de outlier passando NORMALIZED_DIR (entrada) e PROCESSED_DIR (saída)."""
    env_vars = {
        **os.environ,
        "DATA_DIR":        str(NORMALIZED_DIR),  # scripts leem os dados normalizados
        "PROCESSED_DIR":   str(PROCESSED_DIR),
        "REPORTS_DIR":     str(REPORTS_DIR),
        "EXECUTION_DATE":  context["ds"],
    }
    cmd = ["python", str(script_path)]
    logger.info("Executando: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env_vars)

    if result.stdout:
        logger.info("[stdout]\n%s", result.stdout)
    if result.stderr:
        logger.warning("[stderr]\n%s", result.stderr)
    if result.returncode != 0:
        raise AirflowFailException(
            f"'{script_path.name}' falhou com código {result.returncode}."
        )
    logger.info("✔ '%s' concluído.", script_path.name)

def run_isolation_forest(**context) -> None:
    _run_script(ISOLATION_FOREST_SCRIPT, context)

def run_lof(**context) -> None:
    _run_script(LOF_SCRIPT, context)

def run_zscore(**context) -> None:
    _run_script(ZSCORE_SCRIPT, context)

def run_iqr(**context) -> None:
    _run_script(IQR_SCRIPT, context)

def run_knn(**context) -> None:
    _run_script(KNN_SCRIPT, context)
# ---------------------------------------------------------------------------
# Task ML — Random Forest em cada dataset processado
# ---------------------------------------------------------------------------

def _encontrar_arquivo(metodo_sufixo: str) -> Path | None:
    """Retorna o arquivo processado mais recente para o método informado."""
    candidates = (
        list(PROCESSED_DIR.glob(f"*{metodo_sufixo}*.parquet"))
        + list(PROCESSED_DIR.glob(f"*{metodo_sufixo}*.csv"))
    )
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _treinar_e_avaliar(df, metodo: str) -> dict:
    """
    Treina um Random Forest Regressor e retorna métricas de regressão:
    MAE, RMSE e R².
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    colunas_necessarias = FEATURES + [COLUNA_ALVO]
    colunas_presentes   = [c for c in colunas_necessarias if c in df.columns]
    features_presentes  = [c for c in FEATURES if c in df.columns]

    if COLUNA_ALVO not in df.columns:
        raise AirflowFailException(
            f"[{metodo}] Coluna alvo '{COLUNA_ALVO}' não encontrada no dataset."
        )
    if not features_presentes:
        raise AirflowFailException(
            f"[{metodo}] Nenhuma feature encontrada no dataset."
        )

    df_model = df[colunas_presentes].dropna()

    if len(df_model) < 50:
        raise AirflowFailException(
            f"[{metodo}] Dataset com apenas {len(df_model)} linhas após remover NaN — insuficiente para treinar."
        )

    X = df_model[features_presentes]
    y = df_model[COLUNA_ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    # Importância das features
    importancias = dict(zip(features_presentes, reg.feature_importances_.round(4).tolist()))

    logger.info(
        "\n[%s] Métricas de Regressão:\n"
        "  MAE  : %.4f °C\n"
        "  RMSE : %.4f °C\n"
        "  R²   : %.4f",
        metodo.upper(), mae, rmse, r2,
    )
    logger.info("[%s] Importância das features: %s", metodo.upper(), importancias)

    return {
        "metodo":            metodo,
        "n_amostras_treino": len(X_train),
        "n_amostras_teste":  len(X_test),
        "features_usadas":   features_presentes,
        "MAE":               round(mae, 4),
        "RMSE":              round(rmse, 4),
        "R2":                round(r2, 4),
        "feature_importance": importancias,
    }


def ml_random_forest(**context) -> None:
    """
    Treina um Random Forest para cada dataset processado (IQR, Z-Score, LOF,
    Isolation Forest) e gera um relatório comparativo de métricas.

    Alvo     : Temperatura do Ar a 2m  →  Frio / Ameno / Quente
    Features : Umidade, Vento, Direção do Vento, Fluxo de Calor
    Métricas : Accuracy, Precision, Recall, F1-Score
    """
    import json
    import pandas as pd

    resultados = []
    erros      = []

    for task_id, sufixo in METODOS.items():
        arquivo = _encontrar_arquivo(sufixo)

        if arquivo is None:
            msg = f"Arquivo processado não encontrado para método '{sufixo}' em {PROCESSED_DIR}"
            logger.warning("⚠ %s", msg)
            erros.append({"metodo": sufixo, "erro": msg})
            continue

        logger.info("── Treinando com dataset [%s]: %s", sufixo, arquivo)

        df = (
            pd.read_parquet(arquivo)
            if arquivo.suffix == ".parquet"
            else pd.read_csv(arquivo)
        )

        try:
            metricas = _treinar_e_avaliar(df, metodo=sufixo)
            resultados.append(metricas)
        except Exception as exc:
            msg = str(exc)
            logger.error("[%s] Falhou: %s", sufixo, msg)
            erros.append({"metodo": sufixo, "erro": msg})

    if not resultados:
        raise AirflowFailException("Nenhum modelo pôde ser treinado. Verifique os datasets processados.")

    # ── Relatório comparativo ─────────────────────────────────────────────────
    logger.info(
        "\n"
        "╔═══════════════════════════════════════════════════════╗\n"
        "║      COMPARATIVO DE MÉTRICAS — RANDOM FOREST          ║\n"
        "║      Alvo: Temperatura do Ar a 2m (regressão)         ║\n"
        "╠═══════════════════════════════════════════════════════╣\n"
        "║ %-20s %10s %10s %8s ║\n"
        "╠═══════════════════════════════════════════════════════╣",
        "Método", "MAE (°C)", "RMSE (°C)", "R²",
    )
    for r in resultados:
        logger.info(
            "║ %-20s %10.4f %10.4f %8.4f ║",
            r["metodo"], r["MAE"], r["RMSE"], r["R2"],
        )
    logger.info(
        "╚═══════════════════════════════════════════════════════╝"
    )

    # Salva JSON com todos os resultados
    report = {
        "execution_date": context["ds"],
        "alvo":           COLUNA_ALVO,
        "modelo":         "RandomForestRegressor",
        "metricas":       ["MAE", "RMSE", "R2"],
        "resultados":     resultados,
        "erros":          erros,
    }
    report_path = REPORTS_DIR / f"ml_report_{context['ds']}.json"
    report_path.write_text(
        __import__("json").dumps(report, indent=2, ensure_ascii=False)
    )
    logger.info("✔ Relatório ML salvo em: %s", report_path)

    context["ti"].xcom_push(key="ml_report_path", value=str(report_path))
    context["ti"].xcom_push(key="ml_resultados",  value=resultados)


# ---------------------------------------------------------------------------
# Definição da DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="pipeline_outliers_ml",
    description="Detecção de outliers + Random Forest por dataset processado",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["limpeza", "outliers", "ml", "random-forest"],
    doc_md="""
## Pipeline de Interpolação + Outliers + ML

**Fluxo:**
```
start → verificar_dados → interpolacao ──┬─► IQR              ─┐
                                         ├─► Z-Score           ├─► ml_random_forest → end
                                         ├─► LOF               │
                                         └─► Isolation Forest ─┘
```

**Modelo:** Random Forest (regressão)
**Alvo:** Temperatura do Ar a 2m (valor contínuo em °C)
**Métricas:** MAE · RMSE · R²
    """,
    params={
        "data_dir":        str(DATA_DIR),
        "interpolated_dir": str(INTERPOLATED_DIR),
        "processed_dir":   str(PROCESSED_DIR),
        "reports_dir":     str(REPORTS_DIR),
    },
) as dag:

    start = EmptyOperator(task_id="start")

    verificar = PythonOperator(
        task_id="verificar_dados",
        python_callable=verificar_dados,
        doc_md="Verifica se DATA_DIR tem CSVs.",
    )

    interpolacao = PythonOperator(
        task_id="interpolacao",
        python_callable=run_interpolacao,
        execution_timeout=timedelta(hours=6),  # interpolação é pesada
        doc_md="Interpolação espaciotemporal — lê DATA_DIR, salva CSV em INTERPOLATED_DIR.",
    )
    normalizacao = PythonOperator(
        task_id="normalizacao",
        python_callable=run_normalizacao,
        execution_timeout=timedelta(hours=2),
        doc_md="Normalização — lê INTERPOLATED_DIR, salva CSV em NORMALIZED_DIR.",
    )

    isolation_forest = PythonOperator(
        task_id="isolation_forest",
        python_callable=run_isolation_forest,
        doc_md="Isolation Forest — lê INTERPOLATED_DIR, salva em PROCESSED_DIR.",
    )

    lof = PythonOperator(
        task_id="lof",
        python_callable=run_lof,
        doc_md="Local Outlier Factor — lê INTERPOLATED_DIR, salva em PROCESSED_DIR.",
    )

    zscore = PythonOperator(
        task_id="zscore",
        python_callable=run_zscore,
        doc_md="Z-Score — lê INTERPOLATED_DIR, salva em PROCESSED_DIR.",
    )

    iqr = PythonOperator(
        task_id="IQR",
        python_callable=run_iqr,
        doc_md="IQR — lê INTERPOLATED_DIR, salva em PROCESSED_DIR.",
    )

    knn = PythonOperator(
        task_id="knn",
        python_callable=run_knn,
        doc_md="KNN Imputer — lê INTERPOLATED_DIR, salva em PROCESSED_DIR.",
    )

    ml = PythonOperator(
        task_id="ml_random_forest",
        python_callable=ml_random_forest,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md=(
            "Random Forest Regressor para cada dataset processado.\n"
            "Alvo: Temperatura do Ar a 2m (valor contínuo).\n"
            "Métricas: MAE, RMSE, R²."
        ),
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    #
    #   start → verificar → interpolacao ──┬─► isolation_forest ─┐
    #                                      ├─► lof               ├─► ml_random_forest → end
    #                                      ├─► zscore            │
    #                                      └─► IQR ──────────────┘
    #
    start >> verificar >> interpolacao >> normalizacao >> [isolation_forest, lof, zscore, iqr,knn] >> ml >> end