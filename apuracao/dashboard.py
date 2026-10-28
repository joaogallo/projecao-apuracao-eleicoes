"""Gera o dashboard HTML (slider + gráfico) a partir do JSON exportado por ``apuracao.exportar``.

    uv run python -m apuracao.dashboard
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .cli import RAIZ

CAMPOS = ("alvo", "pct_secoes", "pct_validos", "secoes_apuradas", "locais_apurados", "nivel")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", type=Path, default=RAIZ / "resultados" / "dashboard.json")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "dashboard.html")
    ap.add_argument("--site", type=Path, default=RAIZ / "site" / "index.html",
                    help="cópia publicável (pasta servida pelo Vercel); vazio para não gerar")
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
    if str(a.site):
        a.site.parent.mkdir(parents=True, exist_ok=True)
        a.site.write_text(embrulhar(html))
        print("cópia para o site em", a.site)


def embrulhar(html: str) -> str:
    """Página completa para servir fora do claude.ai (o artifact acrescenta esse esqueleto sozinho)."""
    m = re.search(r"<title>.*?</title>", html, re.S)
    titulo = m.group(0) if m else "<title>Simulador de Apuração 2026</title>"
    corpo = html.replace(titulo, "", 1) if m else html
    return (
        "<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"{titulo}\n"
        "<style>html{color-scheme:light dark}body{margin:0;font:14px system-ui,sans-serif}"
        "img{max-width:100%}[hidden]{display:none!important}</style>\n"
        "</head>\n<body>\n" + corpo + "\n</body>\n</html>\n"
    )


if __name__ == "__main__":
    main()
