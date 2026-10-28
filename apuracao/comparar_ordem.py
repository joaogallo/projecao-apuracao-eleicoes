"""Compara a totalização parcial que cada ordem de apuração produz com a série
oficial do TSE (2º turno de 2026).

    uv run python -m apuracao.comparar_ordem --ordens auto,resultados/ordem_bweb_2026_1t.csv,oficial-2026,realista,aleatoria
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .cli import RAIZ
from .dados import carregar_locais, selecionar_turno
from .oficial import carregar_serie_oficial
from .ordem import descrever_ordem, obter_ordem, resolver_ordem

MARCOS = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def curva_parcial(locais: pl.DataFrame, ordem: np.ndarray, pcts: np.ndarray) -> np.ndarray:
    sec = np.cumsum(locais["n_secoes"].to_numpy()[ordem]) / locais["n_secoes"].sum()
    lula = np.cumsum(locais["v_lula"].to_numpy()[ordem])
    val = np.cumsum(locais["validos"].to_numpy()[ordem])
    idx = np.searchsorted(sec, pcts / 100, side="left").clip(0, len(sec) - 1)
    return 100 * lula[idx] / val[idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordens", default="auto,resultados/ordem_bweb_2026_1t.csv,oficial-2026,realista,aleatoria")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resources", type=Path, default=RAIZ / "resources")
    ap.add_argument("--cache", type=Path, default=RAIZ / ".cache")
    a = ap.parse_args(argv)
    locais = selecionar_turno(carregar_locais(2026, a.resources, a.cache), 2)
    of = carregar_serie_oficial().filter(pl.col("pct_secoes") >= 0.005)
    pcts = 100 * of["pct_secoes"].to_numpy()
    alvo = 100 * of["pct_lula"].to_numpy()
    linhas = {}
    for esp in a.ordens.split(","):
        ordem = obter_ordem(locais, esp, a.seed, turno=2)
        sim = curva_parcial(locais, ordem, pcts)
        rmse = float(np.sqrt(np.mean((sim - alvo) ** 2)))
        mae = float(np.mean(np.abs(sim - alvo)))
        print(f"\n== {esp} -> {Path(resolver_ordem(esp, 2)).stem}: {descrever_ordem(esp, 2)}")
        print(f"   RMSE {rmse:.3f} p.p. | erro absoluto médio {mae:.3f} p.p. (Lula, % válidos, {len(pcts)} versões)")
        linhas[esp] = sim
    print("\n% seções | oficial | " + " | ".join(f"{e[:14]:>14}" for e in linhas))
    for m in MARCOS:
        i = int(np.argmin(np.abs(pcts - m)))
        print(f"{pcts[i]:7.2f}% | {alvo[i]:7.2f} | " + " | ".join(f"{linhas[e][i]:14.2f}" for e in linhas))


if __name__ == "__main__":
    main()
