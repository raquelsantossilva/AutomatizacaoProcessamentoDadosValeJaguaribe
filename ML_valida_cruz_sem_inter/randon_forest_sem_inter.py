import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

# ── Caminhos ──────────────────────────────────────────────────────────────────

pasta_saida = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\ML_valida_cruz_sem_inter")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivos = {
    "isolation_forest": Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_sem_interpo\isolation_forest_sem_inter.csv"),
    "knn":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_sem_interpo\knn_sem_inter.csv"),
    "lof":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_sem_interpo\lof_sem_inter.csv"),
    "svm":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos_sem_interpo\svm_sem_inter.csv"),
}

TARGET   = "Temperatura do Ar a 2m"
FEATURES = [
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
]

# ── Scorers ───────────────────────────────────────────────────────────────────

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    "mae":  make_scorer(mean_absolute_error,  greater_is_better=False),
    "rmse": make_scorer(rmse_scorer,          greater_is_better=False),
    "r2":   make_scorer(r2_score),
}

# ── Roda validação cruzada para cada CSV ─────────────────────────────────────

resultados = {}

for tecnica, caminho in arquivos.items():
    print(f"\nProcessando: {tecnica}...")

    df = pd.read_csv(caminho)
    df = df.dropna(subset=[TARGET] + FEATURES)

    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers)

    mae_mean  = round(float(-np.mean(cv_results["test_mae"])),  4)
    mae_std   = round(float(np.std(cv_results["test_mae"])),    4)
    rmse_mean = round(float(-np.mean(cv_results["test_rmse"])), 4)
    rmse_std  = round(float(np.std(cv_results["test_rmse"])),   4)
    r2_mean   = round(float(np.mean(cv_results["test_r2"])),    4)
    r2_std    = round(float(np.std(cv_results["test_r2"])),     4)

    print(f"  MAE  : {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"  RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"  R²   : {r2_mean:.4f} ± {r2_std:.4f}")

    resultados[tecnica] = {
        "tecnica":      tecnica,
        "n_registros":  len(df),
        "mae_mean":     mae_mean,
        "mae_std":      mae_std,
        "rmse_mean":    rmse_mean,
        "rmse_std":     rmse_std,
        "r2_mean":      r2_mean,
        "r2_std":       r2_std,
    }

# ── Salva um JSON por técnica ─────────────────────────────────────────────────

for tecnica, metrics in resultados.items():
    metrics_file = pasta_saida / f"{tecnica}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSalvo: {metrics_file}")