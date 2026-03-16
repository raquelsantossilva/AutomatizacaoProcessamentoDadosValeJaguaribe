from sklearn.preprocessing import RobustScaler
import json
import os
from pathlib import Path
import pandas as pd

# ── Variáveis recebidas da DAG via env var ────────────────────────────────────

_features_env = os.environ.get("FEATURES")
_alvo_env     = os.environ.get("COLUNA_ALVO")

if _features_env:
    FEATURES = json.loads(_features_env)
    print(f"FEATURES recebidas da DAG: {FEATURES}")
else:
    FEATURES = [
        "Umidade Relativa do Ar Mínima a 2m",
        "Velocidade Máxima do Vento 10m",
        "Direção do Vento 2m",
        "Fluxo de Calor no Solo",
    ]
    print(f"[AVISO] Env var FEATURES não encontrada. Usando fallback: {FEATURES}")

if _alvo_env:
    COLUNA_ALVO = _alvo_env
    print(f"COLUNA_ALVO recebida da DAG: {COLUNA_ALVO}")
else:
    COLUNA_ALVO = "Temperatura do Ar a 2m"
    print(f"[AVISO] Env var COLUNA_ALVO não encontrada. Usando fallback: {COLUNA_ALVO}")

# Normaliza features + alvo (sem duplicatas)
COLUNAS_PARA_NORMALIZAR = list(dict.fromkeys([COLUNA_ALVO] + FEATURES))

# ── Caminhos ──────────────────────────────────────────────────────────────────

interpolated_dir = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/interpolated"))
normalized_dir   = Path(os.environ.get("NORMALIZED_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/Normalized"))
normalized_dir.mkdir(parents=True, exist_ok=True)

# Lê o CSV interpolado mais recente da pasta
arquivos = sorted(interpolated_dir.glob("*.csv"))
if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo .csv encontrado em: {interpolated_dir}")

arquivo_entrada = arquivos[-1]
print(f"\nLendo: {arquivo_entrada}")
dados = pd.read_csv(arquivo_entrada)

# ── Normalização ──────────────────────────────────────────────────────────────

# Filtra só as colunas que existem no dataframe
colunas_presentes = [c for c in COLUNAS_PARA_NORMALIZAR if c in dados.columns]
colunas_ausentes  = [c for c in COLUNAS_PARA_NORMALIZAR if c not in dados.columns]

if colunas_ausentes:
    print(f"[AVISO] Colunas não encontradas no dataset (serão ignoradas): {colunas_ausentes}")

if not colunas_presentes:
    raise ValueError("Nenhuma coluna para normalizar foi encontrada no dataset.")

print(f"Normalizando {len(colunas_presentes)} colunas: {colunas_presentes}")

scaler = RobustScaler()
dados_normalizados = dados.copy()
dados_normalizados[colunas_presentes] = scaler.fit_transform(dados[colunas_presentes])

# ── Exportar ──────────────────────────────────────────────────────────────────

saida = normalized_dir / "dados_normalizados.csv"
dados_normalizados.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")
print(f"Shape: {dados_normalizados.shape}")