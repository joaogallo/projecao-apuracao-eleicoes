"""Linha de comando.

Exemplos::

    uv run apuracao --turno 1 --prior "lula=46,bolsonaro=47"
    uv run apuracao --turno 2 --referencia 2022:2 --prior "lula=50,bolsonaro=50" --ordem realista
    uv run apuracao --turno 2 --referencia 2026:1 --ordem aleatoria --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .dados import carregar_locais, resultado_nacional, selecionar_turno
from .modelo import ConfigModelo, Prior, montar_painel
from .ordem import descrever_ordem, obter_ordem, resolver_ordem
from .simulacao import PONTOS_PADRAO, resumo_erros, simular

RAIZ = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ano", type=int, default=2026, help="eleição a projetar (padrão 2026)")
    p.add_argument("--turno", type=int, default=1, choices=[1, 2])
    p.add_argument("--referencia", default=None,
                   help="eleição de referência no formato ANO:TURNO (padrão: 2022 no mesmo turno). "
                        "Mais de uma, separadas por vírgula (ex. 2022:2,2026:1): a primeira é a base do "
                        "swing e as demais entram como covariáveis")
    p.add_argument("--prior", default=None,
                   help='pesquisa de partida, ex. "lula=46,bolsonaro=47" (%%). Padrão: resultado nacional da referência')
    p.add_argument("--peso-prior", type=float, default=ConfigModelo.peso_prior,
                   help="peso da pesquisa em votos-equivalentes")
    p.add_argument("--lambda-regiao", type=float, default=ConfigModelo.lambda_regiao)
    p.add_argument("--lambda-uf", type=float, default=ConfigModelo.lambda_uf)
    p.add_argument("--ordem", default="auto",
                   help="'auto' (padrão: ordem real dos boletins de urna do turno, se "
                        "resultados/ordem_bweb_2026_<turno>t.csv existir, senão 'oficial-2026'), "
                        "'oficial-2026', 'realista', 'aleatoria' ou caminho "
                        "de CSV (uf,cd_municipio,nr_local,ordem)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pontos", default=None,
                   help="frações de seções apuradas em que projetar, ex. '0.01,0.022,0.05,0.1,0.5,1'")
    p.add_argument("--sem-ponderada", action="store_true", help="não calcular a parcial ponderada por município")
    p.add_argument("--resources", type=Path, default=RAIZ / "resources")
    p.add_argument("--cache", type=Path, default=RAIZ / ".cache")
    p.add_argument("--saida", type=Path, default=RAIZ / "resultados")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    refs = [tuple(int(x) for x in r.split(":")) for r in a.referencia.split(",")] if a.referencia else [(2022, a.turno)]
    ano_ref, turno_ref = refs[0]

    atual = selecionar_turno(carregar_locais(a.ano, a.resources, a.cache), a.turno)
    referencia = selecionar_turno(carregar_locais(ano_ref, a.resources, a.cache), turno_ref)
    extras = [selecionar_turno(carregar_locais(y, a.resources, a.cache), t) for y, t in refs[1:]]

    cfg = ConfigModelo(peso_prior=a.peso_prior, lambda_regiao=a.lambda_regiao, lambda_uf=a.lambda_uf,
                       projetar_outros=(a.turno == 1))
    prior = Prior.de_texto(a.prior) if a.prior else Prior(resultado_nacional(referencia))
    pontos = [float(x) for x in a.pontos.split(",")] if a.pontos else PONTOS_PADRAO

    painel = montar_painel(atual, referencia, cfg, extras)
    ordem = obter_ordem(painel.atual, a.ordem, a.seed, a.turno)

    ref_txt = " + ".join(f"{y}:{t}" for y, t in refs)
    print(f"Eleição {a.ano} turno {a.turno} | referência {ref_txt} | seed={a.seed}")
    print("Ordem de apuração:", descrever_ordem(a.ordem, a.turno))
    print(f"Locais: {painel.atual.height} ({painel.casado.mean():.1%} casados com a referência) | "
          f"prior: " + ", ".join(f"{b}={100*v:.1f}%" for b, v in prior.shares.items()))
    res = simular(painel, ordem, prior, cfg, pontos, com_ponderada=not a.sem_ponderada)

    print("\nResultado final real (votos válidos): " + ", ".join(f"{b}={100*v:.2f}%" for b, v in res.final.items()))
    print("\nErro absoluto médio (p.p.) por faixa de apuração:")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(resumo_erros(res.tabela, cfg.blocos))

    a.saida.mkdir(parents=True, exist_ok=True)
    nome = f"simulacao_{a.ano}t{a.turno}_ref{'-'.join(f'{y}t{t}' for y, t in refs)}_{Path(resolver_ordem(a.ordem, a.turno)).stem}_seed{a.seed}.csv"
    res.tabela.write_csv(a.saida / nome)
    print(f"\nTabela salva em {a.saida / nome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
