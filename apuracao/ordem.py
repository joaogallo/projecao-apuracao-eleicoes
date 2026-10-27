"""Ordem em que os locais de votação são apurados.

O TSE não publica o horário de totalização de cada seção neste arquivo, então a
ordem precisa ser simulada ou fornecida externamente (por exemplo, reconstruída
a partir dos boletins de urna, como fez o autor do modelo original).

Modos disponíveis:

* ``aleatoria``: ordem uniformemente aleatória (amostra representativa; é o
  cenário em que a projeção ingênua funcionaria bem e serve de controle).
* ``realista``: reproduz o viés conhecido das apurações brasileiras, em que
  municípios pequenos e as regiões Sul/Sudeste/Centro-Oeste totalizam antes das
  capitais e do Nordeste. Isso faz as primeiras urnas não serem representativas,
  exatamente o problema que o modelo de swing resolve.
* ``oficial-2026`` e ``oficial-2022``: ordens calibradas (deslocamento por UF +
  porte do município + ruído) para que a parcial simulada reproduza a curva
  oficial "% do candidato x % de urnas apuradas" do 2º turno de cada ano, tal
  como divulgada pelo TSE/g1. Ver ``calibrar_ordem.py``.
* arquivo CSV com colunas ``uf, cd_municipio, nr_local, ordem`` (ordem crescente
  ou timestamp), para usar uma ordem real, por exemplo reconstruída dos horários
  de recebimento dos boletins de urna (arquivos "bweb" do TSE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .dados import CHAVE_LOCAL

# Quanto antes (negativo) ou depois (positivo) cada região tende a totalizar.
_OFFSET_REGIAO = {"EXT": -1.2, "S": -0.6, "SE": -0.4, "CO": -0.3, "N": 0.2, "NE": 0.6}
_PESO_PORTE = 0.8   # municípios maiores demoram mais (z-score do log do eleitorado)
_DESVIO_RUIDO = 1.0


# Parâmetros calibrados por ``calibrar_ordem.py`` contra a curva oficial do 2º turno.
CALIBRADAS = {
    "oficial-2026": {  # alvo: série oficial versão a versão do TSE (363 versões), RMSE 0,09 p.p.
        "offset_uf": {"AC": 0.356, "AL": -0.75, "AM": 1.5, "AP": -2.513, "BA": 0.6, "CE": 1.012, "DF": 1.744,
                      "ES": 2.175, "GO": -0.15, "MA": 1.012, "MG": -1.2, "MS": -0.6, "MT": -0.112, "PA": 1.481,
                      "PB": -0.6, "PE": 0.6, "PI": 0.937, "PR": 0.0, "RJ": 1.106, "RN": -0.6, "RO": -0.525,
                      "RR": 2.025, "RS": 0.319, "SC": 0.019, "SE": 0.994, "SP": 0.0, "TO": -0.75, "ZZ": -1.181},
        "w_porte": 0.2,
    },
    "oficial-2022": {
        "offset_uf": {"AC": 0.769, "AL": 0.075, "AM": 0.094, "AP": 1.387, "BA": 0.619, "CE": 0.094, "DF": -0.469,
                      "ES": 0.544, "GO": 0.731, "MA": 0.056, "MG": 1.106, "MS": -1.2, "MT": -0.187, "PA": 0.244,
                      "PB": 0.769, "PE": 0.056, "PI": 0.544, "PR": 0.581, "RJ": -0.037, "RN": -0.244, "RO": -0.731,
                      "RR": -0.619, "RS": 0.544, "SC": -0.6, "SE": -0.112, "SP": -0.619, "TO": 0.562, "ZZ": 0.731},
        "w_porte": 0.219,
    },
}


def _z_porte(locais: pl.DataFrame) -> np.ndarray:
    porte_mun = locais.group_by("cd_municipio", "uf").agg(pl.col("n_secoes").sum().alias("secoes_mun"))
    df = locais.select("uf", "cd_municipio").join(porte_mun, on=["cd_municipio", "uf"], how="left")
    log_porte = np.log(df["secoes_mun"].to_numpy().astype(float))
    return (log_porte - log_porte.mean()) / log_porte.std()


def ordem_simulada(locais: pl.DataFrame, modo: str, seed: int) -> np.ndarray:
    """Retorna os índices (linhas de ``locais``) na ordem em que serão apurados."""
    rng = np.random.default_rng(seed)
    n = locais.height
    if modo == "aleatoria":
        return rng.permutation(n)
    if modo in CALIBRADAS:
        par = CALIBRADAS[modo]
        offset = locais["uf"].replace_strict(par["offset_uf"], default=0.0).to_numpy().astype(float)
        score = offset + par["w_porte"] * _z_porte(locais) + rng.normal(0.0, 1.0, n)
        return np.argsort(score, kind="stable")
    if modo != "realista":
        raise ValueError(f"modo de ordem desconhecido: {modo}")

    offset = locais["regiao"].replace_strict(_OFFSET_REGIAO, default=0.0).to_numpy().astype(float)
    score = _PESO_PORTE * _z_porte(locais) + offset + rng.normal(0.0, _DESVIO_RUIDO, n)
    return np.argsort(score, kind="stable")


def ordem_de_arquivo(locais: pl.DataFrame, caminho: Path, seed: int) -> np.ndarray:
    """Ordem vinda de um CSV ``uf, cd_municipio, nr_local, ordem``.

    Locais ausentes do arquivo vão para o fim, em ordem aleatória.
    """
    ext = pl.read_csv(caminho, schema_overrides={"cd_municipio": pl.Int32, "nr_local": pl.Int32})
    ext = ext.select(CHAVE_LOCAL + ["ordem"]).with_columns(pl.col("ordem").cast(pl.Float64))
    df = locais.select(CHAVE_LOCAL).with_row_index("idx").join(ext, on=CHAVE_LOCAL, how="left")
    rng = np.random.default_rng(seed)
    ordem = df["ordem"].fill_null(float("inf")).to_numpy()
    desempate = rng.random(locais.height)
    return np.lexsort((desempate, ordem))


# Ordem real reconstruída dos boletins de urna (gerada por ``apuracao.bweb``).
ORDEM_REAL_PADRAO = Path(__file__).resolve().parent.parent / "resultados" / "ordem_bweb_2026_1t.csv"


def resolver_ordem(especificacao: str) -> str:
    """``auto`` vira a ordem real dos BUs quando o arquivo existe, senão ``oficial-2026``."""
    if especificacao == "auto":
        return str(ORDEM_REAL_PADRAO) if ORDEM_REAL_PADRAO.exists() else "oficial-2026"
    return especificacao


def descrever_ordem(especificacao: str) -> str:
    e = resolver_ordem(especificacao)
    if Path(e).suffix.lower() == ".csv":
        return ("ordem real: horário de recebimento dos boletins de urna do 1º turno de 2026 "
                "(mediana das seções de cada local de votação)")
    if e in CALIBRADAS:
        return (f"ordem simulada, calibrada para reproduzir a curva oficial de totalização do 2º turno de {e[-4:]} "
                "(deslocamento por UF + porte do município + ruído)")
    if e == "realista":
        return "ordem simulada estilizada: municípios pequenos e Sul/Sudeste/Centro-Oeste antes das capitais e do Nordeste"
    return "ordem aleatória (amostra representativa)"


def obter_ordem(locais: pl.DataFrame, especificacao: str, seed: int) -> np.ndarray:
    e = resolver_ordem(especificacao)
    p = Path(e)
    if p.suffix.lower() == ".csv":
        if not p.exists():
            raise FileNotFoundError(p)
        return ordem_de_arquivo(locais, p, seed)
    return ordem_simulada(locais, e, seed)
