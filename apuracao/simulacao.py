"""Simulação da apuração: percorre a ordem de totalização e registra, em cada
ponto de controle, a parcial ingênua e a projeção do modelo contra o resultado final."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .dados import BLOCOS, resultado_nacional
from .modelo import ConfigModelo, Painel, Prior, parcial_ingenua, parcial_ponderada_municipio, projetar

PONTOS_PADRAO = [0.005, 0.01, 0.02, 0.022, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
                 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]


@dataclass
class ResultadoSimulacao:
    tabela: pl.DataFrame
    final: dict[str, float]


def simular(painel: Painel, ordem: np.ndarray, prior: Prior, cfg: ConfigModelo,
            pontos: list[float] = PONTOS_PADRAO, com_ponderada: bool = True,
            verbose: bool = True) -> ResultadoSimulacao:
    final = resultado_nacional(painel.atual)
    n = painel.n_secoes.size
    secoes_acum = np.cumsum(painel.n_secoes[ordem])
    total_secoes = secoes_acum[-1]
    linhas = []
    apurado = np.zeros(n, dtype=bool)
    pos = 0
    if verbose:
        _cabecalho(cfg.blocos)
    for alvo in pontos:
        if alvo <= 0:
            k = 0
        else:
            k = int(np.searchsorted(secoes_acum, alvo * total_secoes, side="left")) + 1
            k = min(max(k, 1), n)
        apurado[ordem[pos:k]] = True
        pos = k
        proj = projetar(painel, apurado, prior, cfg)
        ing = parcial_ingenua(painel, apurado, cfg.blocos)
        pond = parcial_ponderada_municipio(painel, apurado, cfg.blocos) if com_ponderada else {}
        linha = {
            "alvo": alvo,
            "pct_secoes": proj.pct_secoes,
            "pct_validos": proj.pct_validos,
            "secoes_apuradas": int(painel.n_secoes[apurado].sum()),
            "locais_apurados": int(apurado.sum()),
            "nivel": proj.nivel,
        }
        for b in cfg.blocos:
            linha[f"votos_{b}"] = float(painel.v_atual[apurado, cfg.blocos.index(b)].sum())
            linha[f"proj_votos_{b}"] = proj.votos[b]
            linha[f"parcial_{b}"] = ing[b]
            if com_ponderada:
                linha[f"pond_mun_{b}"] = pond[b]
            linha[f"proj_{b}"] = proj.shares[b]
            linha[f"final_{b}"] = final[b]
            linha[f"erro_parcial_{b}"] = ing[b] - final[b]
            linha[f"erro_proj_{b}"] = proj.shares[b] - final[b]
        linhas.append(linha)
        if verbose:
            _linha(linha, cfg.blocos)
    return ResultadoSimulacao(tabela=pl.DataFrame(linhas), final=final)


def _cabecalho(blocos):
    cols = ["%seções", "%votos", "nível"]
    for b in blocos:
        cols += [f"parc.{b[:4]}", f"proj.{b[:4]}", f"final.{b[:4]}"]
    print(" | ".join(f"{c:>10}" for c in cols))


def _linha(l, blocos):
    vals = [f"{100*l['pct_secoes']:9.2f}%", f"{100*l['pct_validos']:9.2f}%", f"{l['nivel']:>10}"]
    for b in blocos:
        vals += [f"{100*l[f'parcial_{b}']:9.2f}%", f"{100*l[f'proj_{b}']:9.2f}%", f"{100*l[f'final_{b}']:9.2f}%"]
    print(" | ".join(vals))


def resumo_erros(tabela: pl.DataFrame, blocos=BLOCOS) -> pl.DataFrame:
    """Erro absoluto médio (p.p.) da parcial ingênua e da projeção, por faixa de apuração."""
    faixas = [("<=5%", 0.0, 0.05), ("5-20%", 0.05, 0.20), ("20-50%", 0.20, 0.50), (">50%", 0.50, 1.01)]
    linhas = []
    for nome, a, b in faixas:
        t = tabela.filter((pl.col("pct_secoes") > a) & (pl.col("pct_secoes") <= b))
        if t.height == 0:
            continue
        linha = {"faixa": nome, "n_pontos": t.height}
        for bl in blocos:
            linha[f"mae_parcial_{bl}"] = 100 * t[f"erro_parcial_{bl}"].abs().mean()
            linha[f"mae_proj_{bl}"] = 100 * t[f"erro_proj_{bl}"].abs().mean()
        linhas.append(linha)
    return pl.DataFrame(linhas)
