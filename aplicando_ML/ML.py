import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Caminhos ──────────────────────────────────────────────────────────────────

pasta_saida = Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\aplicando_ML")
pasta_saida.mkdir(parents=True, exist_ok=True)

arquivos = {
    "isolation_forest": Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos\isolation_forest.csv"),
    "knn":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos\knn.csv"),
    "lof":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos\lof.csv"),
    "svm":              Path(r"C:\Users\12bea\OneDrive\Documentos\SBBD_algoritms\algoritmos\svm.csv"),
}

TARGET   = "Temperatura do Ar a 2m"
FEATURES = [
    "Umidade Relativa do Ar Mínima a 2m",
    "Velocidade Máxima do Vento 10m",
    "Fluxo de Calor no Solo",
]

# ── Roda Random Forest para cada CSV ─────────────────────────────────────────

resultados = {}

for tecnica, caminho in arquivos.items():
    print(f"\nProcessando: {tecnica}...")

    df = pd.read_csv(caminho)

    # remove linhas com nulo no target ou nas features
    df = df.dropna(subset=[TARGET] + FEATURES)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    resultados[tecnica] = { 
        "tecnica":     tecnica,
        "n_registros": len(df),
        "mae":         round(mae,  4),
        "rmse":        round(rmse, 4),
        "r2":          round(r2,   4),
    }
# ── Salva comparação em JSON ──────────────────────────────────────────────────

for tecnica, metrics in resultados.items():
    metrics_file = pasta_saida / f"{tecnica}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Salvo: {metrics_file}")