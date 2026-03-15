from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pykrige.ok import OrdinaryKriging

# ── Configurações ─────────────────────────────────────────────────────────────

pasta          = Path(os.environ.get("DATA_DIR", "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/dados"))
dados_metadata = "/home/raquel/programacao/estudos/sbbdGRUPO/airflow/stations_metadata.csv"

interpolated_dir = Path(os.environ.get("INTERPOLATED_DIR", "."))
interpolated_dir.mkdir(parents=True, exist_ok=True)

arquivos_vale_jaguaribe = [
    "station_4711_data.csv", "station_5727_data.csv",
    "station_20_data.csv",   "station_59_data.csv",
    "station_21_data.csv",   "station_79_data.csv",
    "station_62_data.csv",   "station_35857_data.csv",
    "station_35742_data.csv","station_24_data.csv",
    "station_6_data.csv",    "station_35855_data.csv"
]

# Só as variáveis usadas no modelo
VARIAVEIS = [
    "Temperatura do Ar a 2m",
    "Umidade Relativa do Ar a 2m",
    "Velocidade Máxima do Vento 10m",
    "Direção do Vento 10m",
    "Fluxo de Calor no Solo",
]

MIN_PONTOS_OK = 3  # mínimo para Krigagem ser válida

# ── Carregamento ──────────────────────────────────────────────────────────────

metadata  = pd.read_csv(dados_metadata)
lista_dfs = []

for f in arquivos_vale_jaguaribe:
    caminho    = pasta / f
    df_temp    = pd.read_csv(caminho)
    id_estacao = re.search(r'\d+', f).group()
    df_temp["id"] = int(id_estacao)
    lista_dfs.append(df_temp)

df = pd.concat(lista_dfs, ignore_index=True)
df = df.merge(metadata, on="id", how="left")

df["data"] = pd.to_datetime(df["data"])
df["ano"]  = df["data"].dt.year
df["mes"]  = df["data"].dt.month
df["dia"]  = df["data"].dt.day
df["hora"] = df["data"].dt.hour

print(f"Dados carregados: {len(df):,} linhas | {df['id'].nunique()} estações")

# ── Funções de interpolação ───────────────────────────────────────────────────

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


def krigagem_batch(lons_ref, lats_ref, lons_src, lats_src, vals_src):
    """
    Krigagem Ordinária para múltiplos pontos alvo em uma única chamada.
    Ganho principal: o variograma é ajustado UMA vez por timestamp,
    não uma vez por ponto NaN.
    """
    try:
        ok = OrdinaryKriging(
            x=lons_src,
            y=lats_src,
            z=vals_src,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
        z_pred, _ = ok.execute("points", lons_ref, lats_ref)
        return np.array(z_pred, dtype=float)
    except Exception:
        return np.full(len(lons_ref), np.nan)


def interpolar_batch(lons_ref, lats_ref, lons_src, lats_src, vals_src):
    """
    Krigagem com fallback IDW:
    - Se pontos suficientes (>= MIN_PONTOS_OK): tenta Krigagem, IDW nos que falharem
    - Se poucos pontos (1 ou 2): usa IDW diretamente (melhor que deixar NaN)
    """
    if len(vals_src) < MIN_PONTOS_OK:
        # poucos pontos — IDW direto, sem tentar Krigagem
        return idw_batch(lons_ref, lats_ref, lons_src, lats_src, vals_src)

    resultados = krigagem_batch(lons_ref, lats_ref, lons_src, lats_src, vals_src)

    mask_nan = np.isnan(resultados)
    if mask_nan.any():
        idw_vals = idw_batch(
            lons_ref[mask_nan], lats_ref[mask_nan],
            lons_src, lats_src, vals_src,
        )
        resultados[mask_nan] = idw_vals

    return resultados


# ── Preenchimento por timestamp (batch) ───────────────────────────────────────

def preencher_variavel(df: pd.DataFrame, variavel: str) -> pd.Series:
    valores  = df[variavel].copy()
    mask_nan = df[variavel].isna()

    if not mask_nan.any():
        return valores

    # ── Nível 1: mesma data/hora — batch por timestamp ────────────────────────
    df_nan     = df[mask_nan]
    timestamps = df_nan.groupby(["ano", "mes", "dia", "hora"]).groups

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
            continue  # nenhum ponto disponível neste timestamp

        alvo = df.loc[idx_nan]
        pred = interpolar_batch(
            alvo["longitude"].values,  alvo["latitude"].values,
            fontes["longitude"].values, fontes["latitude"].values,
            fontes[variavel].values,
        )
        valores.loc[idx_nan] = pred

    # ── Nível 2: mesmo dia, horas ±1–6h (expandido) ──────────────────────────
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
                dif_c.between(1, 6)    &  # expandido de ±3h para ±6h
                outras[variavel].notna()
            ]
            if len(fontes) == 0:
                continue
            pred = interpolar_batch(
                np.array([lon]), np.array([lat]),
                fontes["longitude"].values, fontes["latitude"].values,
                fontes[variavel].values,
            )
            valores.at[idx] = pred[0]

    # ── Nível 3: mesmo mês/hora, qualquer dia ────────────────────────────────
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
            pred = interpolar_batch(
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
        print(f"[AVISO] Coluna '{variavel}' não encontrada. Pulando.")
        continue

    nan_antes = int(df[variavel].isna().sum())
    print(f"\nInterpolando: {variavel}  ({nan_antes:,} NaNs)")

    coluna_saida     = f"{variavel}_preenchida"
    df[coluna_saida] = preencher_variavel(df, variavel)

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