"""Extrai o horário real de recebimento de cada boletim de urna (BU) dos arquivos
``bweb_<turno>_<UF>_<timestamp>.zip`` do TSE e monta a ordem real de apuração.

Os zips são grandes (SP: 1,4 GB comprimido) porque trazem todos os cargos e
candidatos; só precisamos de uma linha por seção. Por isso o CSV é lido em
streaming por ``unzip -p | cut | sort -u`` e só o resultado (uma linha por seção)
entra em memória.

    uv run python -m apuracao.bweb --pasta resources/bweb_2026_1t --saida resultados/ordem_bweb_2026_1t.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import polars as pl

from .cli import RAIZ

# Posições (1-based) das colunas no bweb 2026: SG_UF, CD_MUNICIPIO, NR_ZONA, NR_SECAO,
# NR_LOCAL_VOTACAO, DT_BU_RECEBIDO. Conferidas pelo cabeçalho em tempo de execução.
COLS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_LOCAL_VOTACAO", "DT_BU_RECEBIDO"]


def secoes_do_zip(zip_path: Path, destino: Path) -> Path:
    """Gera ``destino`` com uma linha por seção: uf;municipio;zona;secao;local;dt_recebido."""
    if destino.exists():
        return destino
    # Lê só o cabeçalho sem descomprimir o arquivo todo.
    head = subprocess.run(f"unzip -p '{zip_path}' '*.csv' | head -1", shell=True, stdout=subprocess.PIPE, check=True)
    header = head.stdout.decode("latin-1").strip().replace('"', "").split(";")
    pos = [header.index(c) + 1 for c in COLS]
    campos = ",".join(str(p) for p in pos)
    # LC_ALL=C: o CSV é Latin-1 e, em locale UTF-8, cut/sort abortam com "Illegal byte sequence".
    cmd = f"unzip -p '{zip_path}' '*.csv' | tail -n +2 | cut -d';' -f{campos} | sort -u > '{destino}'"
    subprocess.run(cmd, shell=True, check=True, env={**os.environ, "LC_ALL": "C"})
    return destino


def carregar_secoes(pasta: Path, cache: Path) -> pl.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    partes = []
    for zp in sorted(pasta.glob("bweb_*.zip")):
        dest = cache / (zp.stem + "_secoes.csv")
        print("extraindo", zp.name, flush=True)
        secoes_do_zip(zp, dest)
        df = pl.read_csv(dest, separator=";", has_header=False, new_columns=COLS, quote_char='"',
                         encoding="latin-1", schema_overrides={"CD_MUNICIPIO": pl.Int32, "NR_ZONA": pl.Int32,
                                                                 "NR_SECAO": pl.Int32, "NR_LOCAL_VOTACAO": pl.Int32})
        partes.append(df)
    sec = pl.concat(partes)
    sec = sec.with_columns(pl.col("DT_BU_RECEBIDO").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("recebido"))
    # Uma seção pode aparecer mais de uma vez (ex.: seções agregadas); fica o primeiro BU recebido.
    sec = sec.group_by("SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO").agg(
        pl.col("NR_LOCAL_VOTACAO").first(), pl.col("recebido").min())
    return sec.rename({"SG_UF": "uf", "CD_MUNICIPIO": "cd_municipio", "NR_ZONA": "nr_zona",
                       "NR_SECAO": "nr_secao", "NR_LOCAL_VOTACAO": "nr_local"})


def ordem_por_local(sec: pl.DataFrame) -> pl.DataFrame:
    """Ordem de cada local = mediana do horário de recebimento dos BUs de suas seções."""
    return (
        sec.filter(pl.col("recebido").is_not_null())
        .group_by("uf", "cd_municipio", "nr_local")
        .agg(pl.col("recebido").median().alias("mediana"), pl.col("recebido").min().alias("primeiro"),
             pl.col("recebido").max().alias("ultimo"), pl.len().alias("n_secoes"))
        .with_columns((pl.col("mediana").dt.epoch("s")).alias("ordem"))
        .sort("ordem")
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", type=Path, default=RAIZ / "resources" / "bweb_2026_1t")
    ap.add_argument("--cache", type=Path, default=RAIZ / ".cache" / "bweb")
    ap.add_argument("--saida", type=Path, default=RAIZ / "resultados" / "ordem_bweb_2026_1t.csv")
    a = ap.parse_args(argv)
    sec = carregar_secoes(a.pasta, a.cache)
    sec.write_parquet(a.cache / (a.saida.stem + "_secoes.parquet"))
    print(f"seções: {sec.height}  sem horário: {sec['recebido'].null_count()}")
    print("recebimento: de", sec["recebido"].min(), "a", sec["recebido"].max())
    loc = ordem_por_local(sec)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    loc.select("uf", "cd_municipio", "nr_local", "ordem", "mediana", "primeiro", "ultimo", "n_secoes").write_csv(a.saida)
    print(f"locais: {loc.height}  salvo em {a.saida}")
    # Curva oficial da apuração (% de seções recebidas por horário), para conferência.
    t = sec.filter(pl.col("recebido").is_not_null()).sort("recebido")
    n = t.height
    for frac in (0.01, 0.022, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        print(f"  {100*frac:5.1f}% das seções recebidas até {t['recebido'][min(int(frac*n), n-1)]}")


if __name__ == "__main__":
    main()
