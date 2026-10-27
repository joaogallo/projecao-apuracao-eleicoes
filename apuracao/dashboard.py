"""Gera o dashboard HTML (slider + gráfico) a partir do JSON exportado por ``apuracao.exportar``.

    uv run python -m apuracao.dashboard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import RAIZ

CAMPOS = ("alvo", "pct_secoes", "pct_validos", "secoes_apuradas", "locais_apurados", "nivel")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, default=RAIZ / "resultados" / "dashboard.json")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "dashboard.html")
    a = ap.parse_args(argv)
    dados = json.loads(a.dados.read_text())
    # Mantém só o que a página usa, com 4 casas decimais nas frações.
    for c in dados["cenarios"]:
        serie = []
        for s in c["serie"]:
            linha = {k: s[k] for k in CAMPOS}
            for b in c["blocos"]:
                for pref in ("votos_", "proj_votos_"):
                    linha[pref + b] = round(s[pref + b])
                for pref in ("parcial_", "proj_"):
                    linha[pref + b] = None if s[pref + b] is None else round(s[pref + b], 5)
            serie.append(linha)
        c["serie"] = serie
    template = (Path(__file__).parent / "dashboard_template.html").read_text()
    html = template.replace("__DADOS__", json.dumps(dados, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    a.saida.write_text(html)
    print("salvo em", a.saida, f"({a.saida.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
