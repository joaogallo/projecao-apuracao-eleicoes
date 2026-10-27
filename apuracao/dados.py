"""Leitura dos arquivos ``votacao_secao_<ANO>_BR.csv`` do TSE e agregação por local de votação.

A unidade de análise é o *local de votação* (escola, prédio), identificado por
``(SG_UF, CD_MUNICIPIO, NR_LOCAL_VOTACAO)``. Seções mudam muito entre eleições
(são criadas, divididas, extintas); locais são bem mais estáveis.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

# Blocos de votação comparáveis entre 2022 e 2026 (polarização PT x anti-PT).
BLOCOS = ("lula", "bolsonaro", "outros")
NR_BLOCO = {13: "lula", 22: "bolsonaro"}
NR_BRANCO, NR_NULO = 95, 96

REGIOES = {
    "N": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "NE": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "CO": ["DF", "GO", "MT", "MS"],
    "SE": ["ES", "MG", "RJ", "SP"],
    "S": ["PR", "RS", "SC"],
    "EXT": ["ZZ"],
}
UF_REGIAO = {uf: reg for reg, ufs in REGIOES.items() for uf in ufs}

CHAVE_LOCAL = ["uf", "cd_municipio", "nr_local"]

_SCHEMA = {
    "NR_TURNO": pl.Int8,
    "CD_MUNICIPIO": pl.Int32,
    "NR_ZONA": pl.Int32,
    "NR_SECAO": pl.Int32,
    "CD_CARGO": pl.Int32,
    "NR_VOTAVEL": pl.Int32,
    "QT_VOTOS": pl.Int32,
    "NR_LOCAL_VOTACAO": pl.Int32,
}
_COLUNAS = [
    "NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NM_MUNICIPIO", "NR_ZONA", "NR_SECAO",
    "CD_CARGO", "NR_VOTAVEL", "QT_VOTOS", "NR_LOCAL_VOTACAO", "NM_LOCAL_VOTACAO",
]


def ler_csv_tse(caminho: Path) -> pl.DataFrame:
    """Lê o CSV bruto do TSE (Latin-1, ';', campos entre aspas)."""
    return pl.read_csv(
        caminho, separator=";", encoding="latin-1", quote_char='"',
        columns=_COLUNAS, schema_overrides=_SCHEMA,
    )


def agregar_locais(bruto: pl.DataFrame) -> pl.DataFrame:
    """Agrega votos de Presidente por (turno, local de votação) e por bloco.

    Colunas de saída: turno, uf, regiao, cd_municipio, nm_municipio, nr_local,
    nm_local, n_secoes, v_lula, v_bolsonaro, v_outros, validos, brancos_nulos, total.
    """
    df = bruto.filter(pl.col("CD_CARGO") == 1).with_columns(
        pl.when(pl.col("NR_VOTAVEL") == NR_BRANCO).then(pl.lit("branco"))
        .when(pl.col("NR_VOTAVEL") == NR_NULO).then(pl.lit("nulo"))
        .when(pl.col("NR_VOTAVEL") == 13).then(pl.lit("lula"))
        .when(pl.col("NR_VOTAVEL") == 22).then(pl.lit("bolsonaro"))
        .otherwise(pl.lit("outros"))
        .alias("bloco")
    )
    chave = ["NR_TURNO", "SG_UF", "CD_MUNICIPIO", "NR_LOCAL_VOTACAO"]
    votos = (
        df.group_by(chave + ["bloco"]).agg(pl.col("QT_VOTOS").sum())
        .pivot(on="bloco", index=chave, values="QT_VOTOS")
        .fill_null(0)
    )
    for b in list(BLOCOS) + ["branco", "nulo"]:
        if b not in votos.columns:
            votos = votos.with_columns(pl.lit(0, dtype=pl.Int64).alias(b))
    meta = df.group_by(chave).agg(
        pl.col("NM_MUNICIPIO").first().alias("nm_municipio"),
        pl.col("NM_LOCAL_VOTACAO").first().alias("nm_local"),
        pl.struct("NR_ZONA", "NR_SECAO").n_unique().cast(pl.Int32).alias("n_secoes"),
    )
    out = meta.join(votos, on=chave).rename({
        "NR_TURNO": "turno", "SG_UF": "uf", "CD_MUNICIPIO": "cd_municipio",
        "NR_LOCAL_VOTACAO": "nr_local",
        "lula": "v_lula", "bolsonaro": "v_bolsonaro", "outros": "v_outros",
    })
    out = out.with_columns(
        pl.col("uf").replace_strict(UF_REGIAO, default="?").alias("regiao"),
        (pl.col("v_lula") + pl.col("v_bolsonaro") + pl.col("v_outros")).alias("validos"),
        (pl.col("branco") + pl.col("nulo")).alias("brancos_nulos"),
    ).with_columns((pl.col("validos") + pl.col("brancos_nulos")).alias("total"))
    cols = ["turno", "uf", "regiao", "cd_municipio", "nm_municipio", "nr_local", "nm_local",
            "n_secoes", "v_lula", "v_bolsonaro", "v_outros", "validos", "brancos_nulos", "total"]
    return out.select(cols).sort("turno", "uf", "cd_municipio", "nr_local")


def carregar_locais(ano: int, pasta_resources: Path, pasta_cache: Path) -> pl.DataFrame:
    """Carrega (com cache em parquet) os votos por local de votação de um ano."""
    pasta_cache.mkdir(parents=True, exist_ok=True)
    cache = pasta_cache / f"locais_{ano}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    csv = pasta_resources / f"votacao_secao_{ano}_BR" / f"votacao_secao_{ano}_BR.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    locais = agregar_locais(ler_csv_tse(csv))
    locais.write_parquet(cache)
    return locais


def selecionar_turno(locais: pl.DataFrame, turno: int) -> pl.DataFrame:
    return locais.filter(pl.col("turno") == turno).drop("turno")


def shares(df: pl.DataFrame, prefixo: str = "s_") -> pl.DataFrame:
    """Acrescenta a fração de votos válidos de cada bloco."""
    return df.with_columns([
        (pl.col(f"v_{b}") / pl.col("validos").clip(lower_bound=1)).alias(f"{prefixo}{b}")
        for b in BLOCOS
    ])


def resultado_nacional(df: pl.DataFrame) -> dict[str, float]:
    """Fração de votos válidos de cada bloco no agregado."""
    tot = df.select([pl.col(f"v_{b}").sum() for b in BLOCOS]).row(0)
    validos = sum(tot)
    return {b: v / validos for b, v in zip(BLOCOS, tot)}
