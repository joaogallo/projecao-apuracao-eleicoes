"""Calibra a ordem simulada de apuração para reproduzir a série oficial da totalização.

O TSE não publica o horário de totalização de cada seção do 2º turno, mas publicou o
boletim nacional versão a versão (``resources/apuracao-brasil/``). Este script ajusta
os parâmetros da ordem simulada (um deslocamento por UF e um peso para o porte do
município, com ruído normal de desvio 1) para que a parcial simulada de Lula siga a
parcial oficial em cada versão.

    uv run python -m apuracao.calibrar_ordem

Imprime o dicionário de parâmetros a ser colado em ``ordem.py`` (``CALIBRADAS``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .cli import RAIZ
from .dados import carregar_locais, selecionar_turno
from .oficial import ARQUIVO_T2_2026, alvo_calibracao, carregar_serie_oficial

UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI",
       "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ"]


class Avaliador:
    def __init__(self, locais: pl.DataFrame, curva: dict, seed: int):
        """``curva``: {"serie": (pcts, alvos, pesos)}, listas de mesmo tamanho."""
        porte = locais.group_by("uf", "cd_municipio").agg(pl.col("n_secoes").sum().alias("secoes_mun"))
        df = locais.join(porte, on=["uf", "cd_municipio"], how="left")
        lp = np.log(df["secoes_mun"].to_numpy().astype(float))
        self.z_porte = (lp - lp.mean()) / lp.std()
        self.uf_idx = np.array([UFS.index(u) for u in df["uf"].to_list()])
        self.n_secoes = df["n_secoes"].to_numpy().astype(float)
        self.v_lula = df["v_lula"].to_numpy().astype(float)
        self.validos = df["validos"].to_numpy().astype(float)
        self.ruido = np.random.default_rng(seed).normal(0, 1, df.height)
        pts, alvo, peso = curva["serie"]
        self.pts, self.alvo, self.peso = np.array(pts), np.array(alvo), np.array(peso)

    def curva(self, theta: np.ndarray) -> np.ndarray:
        offsets, w_porte = theta[:len(UFS)], theta[len(UFS)]
        score = offsets[self.uf_idx] + w_porte * self.z_porte + self.ruido
        ordem = np.argsort(score, kind="stable")
        sec = np.cumsum(self.n_secoes[ordem]) / self.n_secoes.sum()
        lula = np.cumsum(self.v_lula[ordem])
        val = np.cumsum(self.validos[ordem])
        idx = np.searchsorted(sec, self.pts / 100, side="left").clip(0, len(sec) - 1)
        return 100 * lula[idx] / val[idx]

    def erro(self, theta: np.ndarray) -> float:
        return float(np.sum(self.peso * (self.curva(theta) - self.alvo) ** 2))


def otimizar(av: Avaliador, theta0: np.ndarray, rodadas: int = 40, seed: int = 0) -> np.ndarray:
    """Busca por coordenadas com passo decrescente (sem dependências externas)."""
    rng = np.random.default_rng(seed)
    theta, melhor = theta0.copy(), av.erro(theta0)
    passo = 0.6
    for r in range(rodadas):
        melhorou = False
        for i in rng.permutation(len(theta)):
            for sinal in (1, -1):
                cand = theta.copy()
                cand[i] += sinal * passo
                e = av.erro(cand)
                if e < melhor:
                    theta, melhor, melhorou = cand, e, True
                    break
        if not melhorou:
            passo *= 0.5
        if passo < 0.01:
            break
    return theta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eleicao", default="2026:2", choices=["2026:2"])
    ap.add_argument("--serie", type=Path, default=ARQUIVO_T2_2026, help="CSV da série oficial versão a versão")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resources", type=Path, default=RAIZ / "resources")
    ap.add_argument("--cache", type=Path, default=RAIZ / ".cache")
    a = ap.parse_args(argv)
    ano, turno = (int(x) for x in a.eleicao.split(":"))
    locais = selecionar_turno(carregar_locais(ano, a.resources, a.cache), turno)
    curva = {"serie": alvo_calibracao(carregar_serie_oficial(a.serie))}
    print(f"alvo: série oficial do TSE, {len(curva['serie'][0])} versões")
    av = Avaliador(locais, curva, a.seed)
    theta0 = np.zeros(len(UFS) + 1)
    theta0[-1] = 0.8
    theta = otimizar(av, theta0)
    curva = av.curva(theta)
    print(f"erro ponderado: {av.erro(theta):.3f}  (RMSE {np.sqrt(np.average((curva - av.alvo) ** 2, weights=av.peso)):.3f} p.p.)")
    marcos = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for m in marcos:
        i = int(np.argmin(np.abs(av.pts - m)))
        print(f"  {av.pts[i]:6.2f}%  simulado {curva[i]:6.2f}  oficial {av.alvo[i]:6.2f}")
    params = {"offset_uf": {u: round(float(v), 3) for u, v in zip(UFS, theta[:len(UFS)])},
              "w_porte": round(float(theta[-1]), 3)}
    print(json.dumps(params, ensure_ascii=False))


if __name__ == "__main__":
    main()
