from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configurações ─────────────────────────────────────────────────────────────

pasta            = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"))
dados_metadata   = "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/stations_metadata.csv"
interpolated_dir = Path(os.environ.get("INTERPOLATED_DIR", "."))
interpolated_dir.mkdir(parents=True, exist_ok=True)

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

# Interpola todas as features + o alvo (sem duplicatas)
VARIAVEIS = list(dict.fromkeys([COLUNA_ALVO] + FEATURES))

JANELA_MM = 72

# ── Lê todos os CSVs da pasta dinamicamente ───────────────────────────────────

arquivos_csv = sorted(pasta.glob("*.csv"))
if not arquivos_csv:
    raise FileNotFoundError(f"Nenhum arquivo .csv encontrado em: {pasta}")

print(f"\nArquivos encontrados em {pasta}: {len(arquivos_csv)}")
for f in arquivos_csv:
    print(f"  {f.name}")

# ── Carregamento ──────────────────────────────────────────────────────────────

metadata  = pd.read_csv(dados_metadata)
lista_dfs = []

for caminho in arquivos_csv:
    df_temp = pd.read_csv(caminho)
    match   = re.search(r'\d+', caminho.name)
    if match:
        df_temp["id"] = int(match.group())
    else:
        print(f"  [AVISO] Não foi possível extrair id de '{caminho.name}' — pulando.")
        continue
    lista_dfs.append(df_temp)

if not lista_dfs:
    raise ValueError("Nenhum arquivo pôde ser carregado.")

df = pd.concat(lista_dfs, ignore_index=True)
df = df.merge(metadata, on="id", how="left")

df["data"] = pd.to_datetime(df["data"])
df["ano"]  = df["data"].dt.year
df["mes"]  = df["data"].dt.month
df["dia"]  = df["data"].dt.day
df["hora"] = df["data"].dt.hour

print(f"\nDados carregados: {len(df):,} linhas | {df['id'].nunique()} estações")

# ── Funções ───────────────────────────────────────────────────────────────────

def haversine_vec(lat1, lon1, lats, lons):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat = lats - lat1
    dlon = lons - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def idw_batch(lons_ref, lats_ref, lons_src, lats_src, vals_src, p=2):
    """IDW para múltiplos pontos alvo de uma vez."""
    resultados = np.full(len(lons_ref), np.nan)
    for i, (lo, la) in enumerate(zip(lons_ref, lats_ref)):
        dist = haversine_vec(la, lo, lats_src, lons_src)
        dist[dist == 0] = 0.0001
        pesos = 1 / (dist ** p)
        resultados[i] = np.sum(pesos * vals_src) / np.sum(pesos)
    return resultados




def preencher_variavel(df: pd.DataFrame, variavel: str) -> pd.Series:
    valores  = df[variavel].copy()
    mask_nan = df[variavel].isna()

    if not mask_nan.any():
        print(f"  Sem NaNs para preencher.")
        return valores

    # ── Nível 1: mesma data/hora — IDW batch por timestamp ───────────────────
    timestamps = df[mask_nan].groupby(["ano", "mes", "dia", "hora"]).groups

    for (ano, mes, dia, hora), idx_nan in tqdm(
        timestamps.items(),
        desc=f"  N1 {variavel[:22]}",
        unit="ts",
        leave=False,
    ):
        fontes = df[
            (df["ano"]  == ano)  &
            (df["mes"]  == mes)  &
            (df["dia"]  == dia)  &
            (df["hora"] == hora) &
            df[variavel].notna()
        ]
        if len(fontes) == 0:
            continue

        alvo = df.loc[idx_nan]
        pred = idw_batch(
            alvo["longitude"].values,   alvo["latitude"].values,
            fontes["longitude"].values, fontes["latitude"].values,
            fontes[variavel].values,
        )
        valores.loc[idx_nan] = pred

    # ── Nível 2: mesmo dia, horas ±1–3h — IDW ────────────────────────────────
    ainda_nan = valores.isna() & mask_nan
    if ainda_nan.any():
        for idx in tqdm(
            df.index[ainda_nan],
            desc=f"  N2 {variavel[:22]}",
            unit="NaN",
            leave=False,
        ):
            row  = df.loc[idx]
            ano, mes, dia, hora = int(row["ano"]), int(row["mes"]), int(row["dia"]), int(row["hora"])
            lat, lon, id_       = row["latitude"], row["longitude"], row["id"]

            outras = df[df["id"] != id_]
            dif    = np.abs(outras["hora"] - hora)
            dif_c  = np.minimum(dif, 24 - dif)
            fontes = outras[
                (outras["ano"] == ano) &
                (outras["mes"] == mes) &
                (outras["dia"] == dia) &
                dif_c.between(1, 3)    &
                outras[variavel].notna()
            ]
            if len(fontes) == 0:
                continue

            pred = idw_batch(
                np.array([lon]), np.array([lat]),
                fontes["longitude"].values, fontes["latitude"].values,
                fontes[variavel].values,
            )
            valores.at[idx] = pred[0]

    # ── Nível 3: mesmo mês/hora, qualquer dia — IDW ───────────────────────────
    ainda_nan = valores.isna() & mask_nan
    if ainda_nan.any():
        for idx in tqdm(
            df.index[ainda_nan],
            desc=f"  N3 {variavel[:22]}",
            unit="NaN",
            leave=False,
        ):
            row  = df.loc[idx]
            ano, mes, hora, id_ = int(row["ano"]), int(row["mes"]), int(row["hora"]), row["id"]
            lat, lon            = row["latitude"], row["longitude"]

            fontes = df[
                (df["ano"]  == ano)  &
                (df["mes"]  == mes)  &
                (df["hora"] == hora) &
                (df["id"]   != id_)  &
                df[variavel].notna()
            ]
            if len(fontes) == 0:
                continue

            pred = idw_batch(
                np.array([lon]), np.array([lat]),
                fontes["longitude"].values, fontes["latitude"].values,
                fontes[variavel].values,
            )
            valores.at[idx] = pred[0]

    return valores


# ── Loop principal ────────────────────────────────────────────────────────────

relatorio = []

for variavel in VARIAVEIS:
    if variavel not in df.columns:
        print(f"\n[AVISO] Coluna '{variavel}' não encontrada no dataset. Pulando.")
        continue

    nan_antes = int(df[variavel].isna().sum())
    print(f"\nInterpolando: {variavel}  ({nan_antes:,} NaNs)")

    coluna_saida     = f"{variavel}_preenchida"
    df[coluna_saida] = preencher_variavel(df, variavel)
    df[variavel]     = df[coluna_saida]  # sobrescreve a coluna original

    nan_depois  = int(df[coluna_saida].isna().sum())
    preenchidos = nan_antes - nan_depois

    relatorio.append({
        "Variável":    variavel,
        "NaN antes":   nan_antes,
        "NaN depois":  nan_depois,
        "Preenchidos": preenchidos,
        "Taxa (%)":    round(preenchidos / nan_antes * 100, 1) if nan_antes > 0 else 0.0,
    })

# ── Relatório ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("              RELATÓRIO CONSOLIDADO DE PREENCHIMENTO")
print("=" * 65)
print(f"{'Variável':<35} {'Antes':>7} {'Depois':>7} {'Preench.':>9} {'Taxa':>6}")
print("-" * 65)
for r in relatorio:
    print(
        f"{r['Variável']:<35} "
        f"{r['NaN antes']:>7,} "
        f"{r['NaN depois']:>7,} "
        f"{r['Preenchidos']:>9,} "
        f"{r['Taxa (%)']:>5.1f}%"
    )
print("=" * 65)

# ── Exportar ──────────────────────────────────────────────────────────────────

saida = interpolated_dir / "interpolacao_todas_variaveis.csv"
df.to_csv(saida, index=False)
print(f"\nArquivo salvo: {saida}")