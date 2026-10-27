"""Série oficial da totalização nacional, versão a versão, publicada pelo TSE.

Fonte: ``resources/apuracao-brasil/apuracao-brasil-por-versao.csv`` (coletor do
projeto Contagem Direta, que gravou cada versão do arquivo
``br-c0001-e006258-u.json`` do TSE no 2º turno de 2026).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_T2_2026 = RAIZ / "resources" / "apuracao-brasil" / "apuracao-brasil-por-versao.csv"


def carregar_serie_oficial(caminho: Path = ARQUIVO_T2_2026) -> pl.DataFrame:
    """Uma linha por versão com seções apuradas > 0, em ordem cronológica.

    Colunas: hora_tse, secoes_apuradas, pct_secoes, votos_validos, lula, bolsonaro,
    pct_lula, pct_bolsonaro (fração dos votos válidos).
    """
    d = pl.read_csv(caminho, schema_overrides={"hora_tse": pl.Utf8, "data_tse": pl.Utf8})
    d = (
        d.with_columns(pl.col("data_tse").str.to_date("%d/%m/%Y").alias("data"))
        .sort("data", "hora_tse")
        .filter(pl.col("secoes_apuradas") > 0)
    )
    return d.select(
        pl.col("hora_tse").str.slice(0, 5).alias("hora_tse"),
        "secoes_apuradas",
        (pl.col("secoes_apuradas") / pl.col("secoes")).alias("pct_secoes"),
        "votos_validos",
        pl.col("lula_13").alias("lula"),
        pl.col("flavio_22").alias("bolsonaro"),
        (pl.col("lula_13") / pl.col("votos_validos")).alias("pct_lula"),
        (pl.col("flavio_22") / pl.col("votos_validos")).alias("pct_bolsonaro"),
    )


def alvo_calibracao(serie: pl.DataFrame, minimo_pct: float = 0.5) -> tuple[list[float], list[float], list[float]]:
    """Pontos (% seções, % Lula, peso) para calibrar a ordem simulada.

    Versões com menos de ``minimo_pct``% das seções são muito ruidosas (poucas
    urnas) e ficam de fora; as demais têm peso 1.
    """
    s = serie.filter(pl.col("pct_secoes") * 100 >= minimo_pct)
    return (
        [100 * v for v in s["pct_secoes"].to_list()],
        [100 * v for v in s["pct_lula"].to_list()],
        [1.0] * s.height,
    )
