"""Exporta a simulação ponto a ponto (0 a 100% das seções) em JSON para o dashboard.

    uv run python -m apuracao.exportar --saida resultados/dashboard.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .cli import RAIZ
from .dados import carregar_locais, resultado_nacional, selecionar_turno
from .modelo import ConfigModelo, Prior, montar_painel
from .oficial import ARQUIVO_T2_2026, carregar_serie_oficial
from .ordem import descrever_ordem, obter_ordem, resolver_ordem
from .simulacao import simular

CENARIOS = [
    # ordem: "auto" = horário real dos boletins de urna do 1º turno (quando extraído).
    # No 2º turno o TSE ainda não publicou os BUs e a ordem do 1º turno não serve de proxy
    # (a apuração do 2º turno foi muito menos enviesada), então a projeção roda sobre a
    # ordem calibrada na série oficial e a totalização mostrada é a oficial, versão a versão.
    {"id": "t1", "rotulo": "1º turno", "turno": 1, "ref": (2022, 1), "prior": "lula=46,bolsonaro=47",
     "prior_rotulo": "Quaest ajustada: Flávio 47%, Lula 46%", "ordem": "auto"},
    {"id": "t2", "rotulo": "2º turno", "turno": 2, "ref": (2022, 2), "extras": [(2026, 1)],
     "prior": "lula=50,bolsonaro=50", "prior_rotulo": "empate técnico: 50% / 50%", "ordem": "oficial-2026"},
]
NOMES = {
    2026: {"lula": "Lula (13)", "bolsonaro": "Flávio Bolsonaro (22)", "outros": "Outros"},
}


def _limpa(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordem", default=None, help="força a mesma ordem nos dois cenários (padrão: a de cada cenário)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resources", type=Path, default=RAIZ / "resources")
    ap.add_argument("--cache", type=Path, default=RAIZ / ".cache")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "dashboard.json")
    a = ap.parse_args(argv)

    pontos = [0.0] + [i / 100 for i in range(1, 101)]
    saida = {"ano": 2026, "seed": a.seed, "nomes": NOMES[2026], "cenarios": []}
    for c in CENARIOS:
        atual = selecionar_turno(carregar_locais(2026, a.resources, a.cache), c["turno"])
        ref = selecionar_turno(carregar_locais(c["ref"][0], a.resources, a.cache), c["ref"][1])
        extras = [selecionar_turno(carregar_locais(y, a.resources, a.cache), t) for y, t in c.get("extras", [])]
        cfg = ConfigModelo(projetar_outros=(c["turno"] == 1))
        prior = Prior.de_texto(c["prior"])
        painel = montar_painel(atual, ref, cfg, extras)
        esp_ordem = a.ordem or c["ordem"]
        ordem = obter_ordem(painel.atual, esp_ordem, a.seed)
        res = simular(painel, ordem, prior, cfg, pontos, com_ponderada=False, verbose=False)
        final = resultado_nacional(atual)
        blocos = [b for b in cfg.blocos if atual[f"v_{b}"].sum() > 0]
        serie = [{k: _limpa(v) for k, v in linha.items()} for linha in res.tabela.to_dicts()]
        oficial = None
        if c["turno"] == 2 and ARQUIVO_T2_2026.exists():
            of = carregar_serie_oficial()
            oficial = [{"pct_secoes": round(r["pct_secoes"], 5), "hora": r["hora_tse"], "secoes": r["secoes_apuradas"],
                        "validos": r["votos_validos"], "votos_lula": r["lula"], "votos_bolsonaro": r["bolsonaro"],
                        "parcial_lula": round(r["pct_lula"], 5), "parcial_bolsonaro": round(r["pct_bolsonaro"], 5)}
                       for r in of.to_dicts()]
        saida["cenarios"].append({
            "oficial": oficial,
            "ordem": Path(resolver_ordem(esp_ordem)).stem,
            "ordem_descricao": descrever_ordem(esp_ordem),
            **{k: c[k] for k in ("id", "rotulo", "turno", "prior", "prior_rotulo")},
            "referencia": " + ".join(f"{y} ({t}º turno)" for y, t in [c["ref"], *c.get("extras", [])]),
            "blocos": blocos,
            "final": {b: final[b] for b in blocos},
            "final_votos": {b: int(atual[f"v_{b}"].sum()) for b in blocos},
            "total_secoes": int(painel.n_secoes.sum()),
            "total_locais": int(painel.n_secoes.size),
            "serie": serie,
        })
        print(c["rotulo"], "ok:", len(serie), "pontos")
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(saida, ensure_ascii=False))
    print("salvo em", a.saida, f"({a.saida.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
