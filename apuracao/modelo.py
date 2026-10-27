"""Modelo de projeção por *swing* de local de votação.

Para cada bloco ``b`` (lula, bolsonaro, outros) e cada local ``i`` já apurado:

    swing_b,i = share_b,i(eleição atual) - share_b,i(eleição de referência)

O swing é explicado por uma regressão linear ponderada pelos votos do local,
com covariáveis de *perfil* (share do bloco na referência) e *porte* (log do
tamanho do município). A regressão é hierárquica: o ajuste de cada UF é
encolhido (ridge) na direção do ajuste da sua região, que por sua vez é
encolhido na direção do ajuste nacional, que é encolhido na direção do
*prior* (a última pesquisa eleitoral). Com poucas urnas, vale a regra
nacional/pesquisa; conforme os votos chegam, as regras por região e por UF
ganham peso: é a "especialização progressiva" descrita pelo autor.

O swing projetado é somado ao share de referência de cada local pendente, e
os votos válidos esperados do local pendente vêm da referência corrigida pela
razão de comparecimento observada entre os locais já apurados.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .dados import BLOCOS, CHAVE_LOCAL, shares


@dataclass
class ConfigModelo:
    # Peso do prior (pesquisa) em votos-equivalentes: quantos votos apurados são
    # necessários para que os dados pesem tanto quanto a pesquisa.
    peso_prior: float = 3_000_000.0
    # Encolhimento região -> nacional e UF -> região, em votos-equivalentes.
    lambda_regiao: float = 300_000.0
    lambda_uf: float = 150_000.0
    # Encolhimento da razão de comparecimento (votos válidos atual / referência).
    lambda_comparecimento: float = 200_000.0
    # Se True, o share de "outros" também recebe swing; se False, "outros" é
    # projetado por diferença (útil no 2º turno, em que outros = 0).
    projetar_outros: bool = True
    blocos: tuple[str, ...] = field(default_factory=lambda: BLOCOS)


@dataclass
class Prior:
    """Fração de votos válidos esperada para cada bloco (ex.: última pesquisa)."""
    shares: dict[str, float]

    @classmethod
    def de_texto(cls, texto: str) -> "Prior":
        """Ex.: ``lula=46,bolsonaro=47`` (outros = resto). Valores em %."""
        d = {}
        for par in texto.split(","):
            k, v = par.split("=")
            d[k.strip()] = float(v) / 100.0
        if "outros" not in d:
            d["outros"] = max(0.0, 1.0 - sum(d.values()))
        s = sum(d.values())
        return cls({k: v / s for k, v in d.items()})


@dataclass
class Painel:
    """Todos os locais da eleição atual alinhados com a referência.

    ``atual``: votos reais (usados só para o que já foi "apurado").
    ``ref``: perfil de referência de cada local (casado ou imputado pelo município/UF).
    """
    atual: pl.DataFrame
    s_ref: np.ndarray           # (n, nb) share de referência por bloco
    validos_ref: np.ndarray     # (n,) votos válidos esperados a partir da referência
    casado: np.ndarray          # (n,) bool: tem par real na referência
    X: np.ndarray               # (n, nb, p) covariáveis por bloco
    uf: np.ndarray
    regiao: np.ndarray
    v_atual: np.ndarray         # (n, nb) votos reais por bloco (realidade)
    validos_atual: np.ndarray   # (n,)
    n_secoes: np.ndarray        # (n,)


def montar_painel(atual: pl.DataFrame, referencia: pl.DataFrame, cfg: ConfigModelo,
                  extras: Sequence[pl.DataFrame] = ()) -> Painel:
    """Casa os locais da eleição atual com a referência e monta as matrizes.

    ``referencia`` é a eleição-base do swing. ``extras`` são outras eleições
    (ex.: o 1º turno do mesmo ano, quando se projeta o 2º) cujo share de cada
    bloco entra como covariável adicional da regressão: é assim que o autor
    combina o 2º turno de 2022 com o 1º turno de 2026.
    """
    nb = len(cfg.blocos)
    ref = shares(referencia).select(
        CHAVE_LOCAL + ["n_secoes", "validos"] + [f"s_{b}" for b in cfg.blocos]
    ).rename({"n_secoes": "n_secoes_ref", "validos": "validos_ref"})

    # Perfis de fallback: município e UF (para locais novos sem par em 2022).
    agg_cols = [pl.col(f"v_{b}").sum() for b in cfg.blocos] + [
        pl.col("validos").sum(), pl.col("n_secoes").sum()]
    mun = shares(referencia.group_by("uf", "cd_municipio").agg(agg_cols)).with_columns(
        (pl.col("validos") / pl.col("n_secoes")).alias("validos_por_secao_mun")
    ).select(["uf", "cd_municipio", "validos_por_secao_mun"] + [
        pl.col(f"s_{b}").alias(f"s_{b}_mun") for b in cfg.blocos] + [pl.col("validos").alias("validos_mun_ref")])
    ufa = shares(referencia.group_by("uf").agg(agg_cols)).with_columns(
        (pl.col("validos") / pl.col("n_secoes")).alias("validos_por_secao_uf")
    ).select(["uf", "validos_por_secao_uf"] + [pl.col(f"s_{b}").alias(f"s_{b}_uf") for b in cfg.blocos]
             + [pl.col("validos").alias("validos_uf_ref")])

    df = (
        atual.join(ref, on=CHAVE_LOCAL, how="left")
        .join(mun, on=["uf", "cd_municipio"], how="left")
        .join(ufa, on="uf", how="left")
    )
    casado = df["validos_ref"].is_not_null() & (df["validos_ref"] > 0)
    df = df.with_columns(casado.alias("casado"))

    # Share de referência: local casado -> próprio; senão município; senão UF.
    # Município sem votos na referência (ex.: posto no exterior fechado) cai para a UF.
    mun_ok = pl.col("validos_mun_ref").fill_null(0) > 0
    s_ref_exprs = []
    for b in cfg.blocos:
        s_ref_exprs.append(
            pl.when(pl.col("casado")).then(pl.col(f"s_{b}"))
            .when(mun_ok).then(pl.col(f"s_{b}_mun"))
            .otherwise(pl.col(f"s_{b}_uf"))
            .alias(f"sref_{b}")
        )
    # Votos válidos esperados pela referência, ajustados pelo nº de seções atual.
    validos_ref_expr = (
        pl.when(pl.col("casado"))
        .then(pl.col("validos_ref") * pl.col("n_secoes") / pl.col("n_secoes_ref").clip(lower_bound=1))
        .when(mun_ok).then(pl.col("n_secoes") * pl.col("validos_por_secao_mun"))
        .otherwise(pl.col("n_secoes") * pl.col("validos_por_secao_uf"))
        .alias("validos_ref_esp")
    )
    porte_expr = (
        pl.when(mun_ok).then(pl.col("validos_mun_ref")).otherwise(pl.col("validos_uf_ref"))
        .clip(lower_bound=1).log10().alias("log_porte")
    )
    df = df.with_columns(s_ref_exprs + [validos_ref_expr, porte_expr])

    # Referências extras: share de cada bloco (local; senão município; senão UF).
    extra_cols = []
    for k, ex in enumerate(extras):
        exs = shares(ex).select(CHAVE_LOCAL + [pl.col(f"s_{b}").alias(f"x{k}_{b}") for b in cfg.blocos]
                                + [pl.col("validos").alias(f"x{k}_validos")])
        exm = shares(ex.group_by("uf", "cd_municipio").agg(agg_cols)).select(
            ["uf", "cd_municipio"] + [pl.col(f"s_{b}").alias(f"x{k}_{b}_mun") for b in cfg.blocos])
        exu = shares(ex.group_by("uf").agg(agg_cols)).select(
            ["uf"] + [pl.col(f"s_{b}").alias(f"x{k}_{b}_uf") for b in cfg.blocos])
        df = df.join(exs, on=CHAVE_LOCAL, how="left").join(exm, on=["uf", "cd_municipio"], how="left").join(exu, on="uf", how="left")
        ok = pl.col(f"x{k}_validos").fill_null(0) > 0
        for b in cfg.blocos:
            df = df.with_columns(
                pl.when(ok).then(pl.col(f"x{k}_{b}"))
                .otherwise(pl.coalesce(pl.col(f"x{k}_{b}_mun"), pl.col(f"x{k}_{b}_uf")))
                .fill_null(0.0).alias(f"xref{k}_{b}")
            )
            extra_cols.append((k, b))

    s_ref = np.column_stack([df[f"sref_{b}"].to_numpy().astype(float) for b in cfg.blocos])
    validos_ref = df["validos_ref_esp"].to_numpy().astype(float)
    log_porte = df["log_porte"].to_numpy().astype(float)
    w = validos_ref
    # Padroniza covariáveis (média/desvio ponderados) para o ridge fazer sentido.
    def z(x):
        m = np.average(x, weights=w)
        s = np.sqrt(np.average((x - m) ** 2, weights=w)) or 1.0
        return (x - m) / s
    zp = z(log_porte)
    n_extra = len(extras)
    X = np.empty((df.height, nb, 3 + n_extra))
    for j, b in enumerate(cfg.blocos):
        X[:, j, 0] = 1.0
        X[:, j, 1] = z(s_ref[:, j])
        X[:, j, 2] = zp
        for k in range(n_extra):
            # Diferença entre a referência extra e a base: o swing já observado
            # entre as duas eleições anteriores, por local.
            X[:, j, 3 + k] = z(df[f"xref{k}_{b}"].to_numpy().astype(float) - s_ref[:, j])

    v_atual = np.column_stack([df[f"v_{b}"].to_numpy().astype(float) for b in cfg.blocos])
    return Painel(
        atual=df, s_ref=s_ref, validos_ref=validos_ref, casado=casado.to_numpy(), X=X,
        uf=df["uf"].to_numpy(), regiao=df["regiao"].to_numpy(),
        v_atual=v_atual, validos_atual=df["validos"].to_numpy().astype(float),
        n_secoes=df["n_secoes"].to_numpy().astype(float),
    )


def _wls_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, beta_pai: np.ndarray, lam: float) -> np.ndarray:
    """Mínimos quadrados ponderados encolhidos para ``beta_pai`` com força ``lam``."""
    p = X.shape[1]
    A = X.T @ (X * w[:, None]) + lam * np.eye(p)
    b = X.T @ (w * y) + lam * beta_pai
    return np.linalg.solve(A, b)


def _razao_encolhida(num: float, den: float, pai: float, lam: float) -> float:
    return (num + lam * pai) / (den + lam) if (den + lam) > 0 else pai


@dataclass
class Projecao:
    shares: dict[str, float]          # projeção final (fração de votos válidos)
    votos: dict[str, float]           # votos projetados por bloco
    pct_secoes: float                 # fração de seções apuradas
    pct_validos: float                # fração dos votos válidos (reais) já apurados
    nivel: str                        # "prior", "nacional", "regiao", "uf" (nível predominante)


def projetar(painel: Painel, apurado: np.ndarray, prior: Prior, cfg: ConfigModelo) -> Projecao:
    """Projeta o resultado final dado o conjunto de locais já apurados."""
    nb = len(cfg.blocos)
    treino = apurado & painel.casado
    s_atual = painel.v_atual / np.clip(painel.validos_atual, 1, None)[:, None]
    swing = s_atual - painel.s_ref                      # (n, nb)
    w_tr = painel.validos_atual[treino]

    # Prior do intercepto: swing nacional implícito na pesquisa.
    s_ref_nac = (painel.s_ref * painel.validos_ref[:, None]).sum(0) / painel.validos_ref.sum()
    p = painel.X.shape[2]
    beta_prior = np.zeros((nb, p))
    for j, b in enumerate(cfg.blocos):
        beta_prior[j, 0] = prior.shares.get(b, s_ref_nac[j]) - s_ref_nac[j]

    # Ajuste hierárquico: nacional -> região -> UF, por bloco.
    beta_nac = np.zeros((nb, p))
    for j in range(nb):
        beta_nac[j] = _wls_ridge(painel.X[treino, j], swing[treino, j], w_tr, beta_prior[j], cfg.peso_prior)

    regioes = np.unique(painel.regiao)
    ufs = np.unique(painel.uf)
    beta_reg = {}
    for r in regioes:
        m = treino & (painel.regiao == r)
        beta_reg[r] = np.array([
            _wls_ridge(painel.X[m, j], swing[m, j], painel.validos_atual[m], beta_nac[j], cfg.lambda_regiao)
            for j in range(nb)])
    beta_uf = {}
    for u in ufs:
        m = treino & (painel.uf == u)
        r = painel.regiao[painel.uf == u][0]
        beta_uf[u] = np.array([
            _wls_ridge(painel.X[m, j], swing[m, j], painel.validos_atual[m], beta_reg[r][j], cfg.lambda_uf)
            for j in range(nb)])

    # Razão de comparecimento (válidos atual / válidos referência), hierárquica.
    razao_nac = _razao_encolhida(painel.validos_atual[treino].sum(), painel.validos_ref[treino].sum(), 1.0, cfg.lambda_comparecimento)
    razao_reg = {r: _razao_encolhida(painel.validos_atual[treino & (painel.regiao == r)].sum(),
                                     painel.validos_ref[treino & (painel.regiao == r)].sum(),
                                     razao_nac, cfg.lambda_comparecimento) for r in regioes}
    razao_uf = {}
    for u in ufs:
        m = treino & (painel.uf == u)
        r = painel.regiao[painel.uf == u][0]
        razao_uf[u] = _razao_encolhida(painel.validos_atual[m].sum(), painel.validos_ref[m].sum(), razao_reg[r], cfg.lambda_comparecimento)

    # Projeção dos locais pendentes.
    pend = ~apurado
    s_proj = np.zeros((pend.sum(), nb))
    validos_proj = np.zeros(pend.sum())
    idx_pend = np.where(pend)[0]
    for u in ufs:
        sel = painel.uf[idx_pend] == u
        if not sel.any():
            continue
        ii = idx_pend[sel]
        for j in range(nb):
            s_proj[sel, j] = painel.s_ref[ii, j] + painel.X[ii, j] @ beta_uf[u][j]
        validos_proj[sel] = painel.validos_ref[ii] * razao_uf[u]
    s_proj = np.clip(s_proj, 0.0, 1.0)
    # Bloco sem nenhum voto na eleição atual (ex.: "outros" no 2º turno) não existe.
    ativo = painel.v_atual.sum(0) > 0
    s_proj[:, ~ativo] = 0.0
    if not cfg.projetar_outros and "outros" in cfg.blocos and ativo[cfg.blocos.index("outros")]:
        jo = cfg.blocos.index("outros")
        resto = np.delete(s_proj, jo, axis=1).sum(1)
        s_proj[:, jo] = np.clip(1.0 - resto, 0.0, 1.0)
    soma = s_proj.sum(1, keepdims=True)
    s_proj = np.where(soma > 0, s_proj / np.clip(soma, 1e-9, None), ativo / ativo.sum())

    votos = painel.v_atual[apurado].sum(0) + (s_proj * validos_proj[:, None]).sum(0)
    total = votos.sum()
    pct_validos = painel.validos_atual[apurado].sum() / painel.validos_atual.sum()
    pct_secoes = painel.n_secoes[apurado].sum() / painel.n_secoes.sum()
    n_votos_tr = w_tr.sum()
    if n_votos_tr < cfg.peso_prior * 0.1:
        nivel = "prior"
    elif n_votos_tr < cfg.lambda_regiao * len(regioes):
        nivel = "nacional"
    elif n_votos_tr < cfg.lambda_uf * len(ufs):
        nivel = "regiao"
    else:
        nivel = "uf"
    return Projecao(
        shares={b: votos[j] / total for j, b in enumerate(cfg.blocos)},
        votos={b: float(votos[j]) for j, b in enumerate(cfg.blocos)},
        pct_secoes=float(pct_secoes), pct_validos=float(pct_validos), nivel=nivel,
    )


def parcial_ingenua(painel: Painel, apurado: np.ndarray, blocos=BLOCOS) -> dict[str, float]:
    """A "projeção" ingênua: a fração de votos válidos já apurados (o que a TV mostra)."""
    v = painel.v_atual[apurado].sum(0)
    t = v.sum()
    return {b: (v[j] / t if t > 0 else float("nan")) for j, b in enumerate(blocos)}


def parcial_ponderada_municipio(painel: Painel, apurado: np.ndarray, blocos=BLOCOS) -> dict[str, float]:
    """Hipótese descartada pelo autor: parcial de cada município ponderada pelo
    seu peso no eleitorado (município sem urna apurada recebe a parcial nacional).
    Falha porque as primeiras urnas de um município não são amostra dele."""
    df = painel.atual.select("uf", "cd_municipio").with_columns(
        pl.Series("apurado", apurado),
        pl.Series("peso", painel.validos_ref),
        *[pl.Series(f"v_{b}", painel.v_atual[:, j]) for j, b in enumerate(blocos)],
    )
    nac = parcial_ingenua(painel, apurado, blocos)
    g = df.group_by("uf", "cd_municipio").agg(
        pl.col("peso").sum(),
        *[pl.col(f"v_{b}").filter(pl.col("apurado")).sum() for b in blocos],
    )
    v = np.column_stack([g[f"v_{b}"].to_numpy() for b in blocos]).astype(float)
    tot = v.sum(1)
    peso = g["peso"].to_numpy().astype(float)
    share = np.where(tot[:, None] > 0, v / np.clip(tot, 1, None)[:, None],
                     np.array([nac[b] for b in blocos])[None, :])
    agg = (share * peso[:, None]).sum(0) / peso.sum()
    return {b: float(agg[j]) for j, b in enumerate(blocos)}
