from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import TriggerRule


BASE_DIR    = Path(os.getenv("PREPROCESSING_BASE_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow"))
SCRIPTS_DIR = Path(os.getenv("SCRIPTS_DIR",            "/home/raquel/programacao/estudos/sbbdGRUPO/limpeza"))
DATA_DIR    = Path(os.getenv("DATA_DIR",               "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"))

INTERPOLATED_DIR = BASE_DIR / "interpolated"
NORMALIZED_DIR   = BASE_DIR / "Normalized"
PROCESSED_DIR    = BASE_DIR / "processed"
REPORTS_DIR      = BASE_DIR / "reports"

for _dir in (INTERPOLATED_DIR, NORMALIZED_DIR, PROCESSED_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

NORMALIZACAO_SCRIPT = Path(os.getenv("NORMALIZACAO_SCRIPT", "/home/raquel/programacao/estudos/sbbdGRUPO/limpeza/normalizacao.py"))
INTERPOLACAO_SCRIPT = Path(os.getenv("INTERPOLACAO_SCRIPT", "/home/raquel/programacao/estudos/sbbdGRUPO/limpeza/interpolacao/espacotemp.py"))

ISOLATION_FOREST_SCRIPT = SCRIPTS_DIR / "isolation_forest.py"
LOF_SCRIPT              = SCRIPTS_DIR / "lof.py"
KNN_SCRIPT              = SCRIPTS_DIR / "knn.py"
SVM_SCRIPT              = SCRIPTS_DIR / "svm.py"

COLUNA_ALVO = "Temperatura do Ar a 2m"
FEATURES    = [
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
]

METODOS = {
    "lof":              "lof",
    "isolation_forest": "isolation_forest",
    "knn":              "knn",
    "svm":              "svm",
}

default_args = {
    "owner":             "data-engineering",
    "depends_on_past":   False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=60),
}


def verificar_dados(**context) -> None:
    csv_files = list(DATA_DIR.glob("*.csv"))
    context["ti"].xcom_push(key="data_files", value=[str(f) for f in csv_files])


def run_interpolacao(**context) -> None:
    env_vars = {
        **os.environ,
        "DATA_DIR":         str(DATA_DIR),
        "INTERPOLATED_DIR": str(INTERPOLATED_DIR),
        "EXECUTION_DATE":   context["ts"],
    }
    subprocess.run(["python", str(INTERPOLACAO_SCRIPT)], capture_output=True, text=True, env=env_vars)
    csvs = list(INTERPOLATED_DIR.glob("*.csv"))
    context["ti"].xcom_push(key="interpolated_files", value=[str(f) for f in csvs])


def run_normalizacao(**context) -> None:
    env_vars = {
        **os.environ,
        "DATA_DIR":       str(INTERPOLATED_DIR),
        "NORMALIZED_DIR": str(NORMALIZED_DIR),
        "EXECUTION_DATE": context["ts"],
    }
    subprocess.run(["python", str(NORMALIZACAO_SCRIPT)], capture_output=True, text=True, env=env_vars)
    csvs = list(NORMALIZED_DIR.glob("*.csv"))
    context["ti"].xcom_push(key="normalized_files", value=[str(f) for f in csvs])


def _run_script(script_path: Path, context: dict) -> None:
    env_vars = {
        **os.environ,
        "DATA_DIR":       str(NORMALIZED_DIR),
        "PROCESSED_DIR":  str(PROCESSED_DIR),
        "REPORTS_DIR":    str(REPORTS_DIR),
        "EXECUTION_DATE": context["ts"],
    }
    subprocess.run(["python", str(script_path)], capture_output=True, text=True, env=env_vars)


def run_isolation_forest(**context) -> None:
    _run_script(ISOLATION_FOREST_SCRIPT, context)

def run_lof(**context) -> None:
    _run_script(LOF_SCRIPT, context)

def run_knn(**context) -> None:
    _run_script(KNN_SCRIPT, context)

def run_svm(**context) -> None:
    _run_script(SVM_SCRIPT, context)


def _encontrar_arquivo(metodo_sufixo: str) -> Path | None:
    candidates = (
        list(PROCESSED_DIR.glob(f"*{metodo_sufixo}*.parquet"))
        + list(PROCESSED_DIR.glob(f"*{metodo_sufixo}*.csv"))
    )
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _treinar_e_avaliar(df, metodo: str) -> dict | None:
    colunas_presentes  = [c for c in FEATURES + [COLUNA_ALVO] if c in df.columns]
    features_presentes = [c for c in FEATURES if c in df.columns]

    if COLUNA_ALVO not in df.columns or not features_presentes:
        return None

    df_model = df[colunas_presentes].dropna()
    if len(df_model) < 50:
        return None

    X = df_model[features_presentes]
    y = df_model[COLUNA_ALVO]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    return {
        "metodo":             metodo,
        "n_amostras_treino":  len(X_train),
        "n_amostras_teste":   len(X_test),
        "features_usadas":    features_presentes,
        "MAE":                round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE":               round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "R2":                 round(r2_score(y_test, y_pred), 4),
        "feature_importance": dict(zip(features_presentes, reg.feature_importances_.round(4).tolist())),
    }


def ml_random_forest(**context) -> None:
    resultados = []

    for sufixo in METODOS.values():
        arquivo = _encontrar_arquivo(sufixo)
        if arquivo is None:
            continue
        df = pd.read_parquet(arquivo) if arquivo.suffix == ".parquet" else pd.read_csv(arquivo)
        metricas = _treinar_e_avaliar(df, metodo=sufixo)
        if metricas:
            resultados.append(metricas)

    report = {
        "execution_date": context["ts"],
        "alvo":           COLUNA_ALVO,
        "modelo":         "RandomForestRegressor",
        "metricas":       ["MAE", "RMSE", "R2"],
        "resultados":     resultados,
    }

    ts_safe = context["ts"].replace(":", "-").replace("+", "_")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"ml_report_{ts_safe}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    context["ti"].xcom_push(key="ml_report_path", value=str(report_path))
    context["ti"].xcom_push(key="ml_resultados",  value=resultados)


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
start → verificar_dados → interpolacao → normalizacao ──┬─► lof              ─┐
                                                        ├─► isolation_forest  ├─► ml_random_forest → end
                                                        ├─► knn               │
                                                        └─► svm ──────────────┘
```

**Modelo:** Random Forest (regressão)
**Alvo:** Temperatura do Ar a 2m (valor contínuo em °C)
**Métricas:** MAE · RMSE · R²
    """,
    params={
        "data_dir":         str(DATA_DIR),
        "interpolated_dir": str(INTERPOLATED_DIR),
        "processed_dir":    str(PROCESSED_DIR),
        "reports_dir":      str(REPORTS_DIR),
    },
) as dag:

    start = EmptyOperator(task_id="start")

    verificar = PythonOperator(
        task_id="verificar_dados",
        python_callable=verificar_dados,
    )

    interpolacao = PythonOperator(
        task_id="interpolacao",
        python_callable=run_interpolacao,
        execution_timeout=timedelta(hours=6),
    )

    normalizacao = PythonOperator(
        task_id="normalizacao",
        python_callable=run_normalizacao,
        execution_timeout=timedelta(hours=2),
    )

    isolation_forest = PythonOperator(
        task_id="isolation_forest",
        python_callable=run_isolation_forest,
    )

    lof = PythonOperator(
        task_id="lof",
        python_callable=run_lof,
    )

    svm = PythonOperator(
        task_id="svm",
        python_callable=run_svm,
    )

    knn = PythonOperator(
        task_id="knn",
        python_callable=run_knn,
    )

    ml = PythonOperator(
        task_id="ml_random_forest",
        python_callable=ml_random_forest,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    start >> verificar >> interpolacao >> normalizacao >> [isolation_forest, lof, svm, knn] >> ml >> endsq                                                                                                                                                                                                                  