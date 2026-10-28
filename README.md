# Projeção de resultado durante a apuração

Implementação em Python do algoritmo descrito por Rodrigo Mahlmeister em entrevista ao
Estadão e na matéria do NeoFeed
([Como um pesquisador brasileiro criou um algoritmo que simulou e acertou o resultado do 1º turno](https://neofeed.com.br/finde/como-um-pesquisador-brasileiro-criou-um-algoritmo-que-simulou-e-acertou-o-resultado-do-1o-turno/)).

A ideia: em vez de estimar intenção de voto, o modelo parte dos votos **já apurados**,
mede quanto cada local de votação mudou em relação à eleição anterior (o *swing*) e
usa isso para projetar as urnas que ainda faltam.

## Dados

Os arquivos grandes não estão versionados (ver `.gitignore`); para rodar o projeto é
preciso baixá-los do [portal de dados abertos do TSE](https://dadosabertos.tse.jus.br/):

- `resources/votacao_secao_2026_BR/votacao_secao_2026_BR.csv` e
  `resources/votacao_secao_2022_BR/votacao_secao_2022_BR.csv` (1,5 GB cada): conjunto
  "Resultados - Votação por seção eleitoral", arquivo `votacao_secao_<ANO>_BR.zip`.
- `resources/bweb_2026_1t/bweb_1t_<UF>_*.zip` (5,1 GB no total, opcional): conjunto
  "Resultados - 2026 - Boletim de Urna", um zip por UF, usado só para reconstruir a ordem
  real de apuração. O resultado dessa extração já está versionado em
  `resultados/ordem_bweb_2026_1t.csv`.
- A entrevista em vídeo com o autor do modelo e sua transcrição também ficam fora do
  repositório.

O que está versionado: `resources/apuracao-brasil/` (série oficial do 2º turno, versão a
versão), os `leiame.pdf` do TSE e tudo em `resultados/`.

- `resources/votacao_secao_2026_BR/` e `resources/votacao_secao_2022_BR/`: votação por
  seção do TSE (Presidente, 1º e 2º turnos). Leiaute em `leiame.pdf`.
- A unidade de análise é o **local de votação** (`SG_UF`, `CD_MUNICIPIO`,
  `NR_LOCAL_VOTACAO`), que é muito mais estável entre eleições do que a seção.
  93% dos locais de 2026 (97% dos votos) têm par em 2022; os demais recebem o perfil
  do município (ou da UF).
- Os votos são agrupados em três blocos comparáveis entre as eleições:
  `lula` (13), `bolsonaro` (22: Jair em 2022, Flávio em 2026) e `outros`.
  Brancos e nulos ficam fora (as projeções são sobre votos válidos).

## O algoritmo (`apuracao/modelo.py`)

1. **Ponto de partida (prior)**: a última pesquisa (ex.: `lula=46,bolsonaro=47`).
   Sem nada apurado, a projeção é a pesquisa.
2. **Swing por local**: para cada local já apurado e casado com a referência,
   `swing_b = share_b(2026) − share_b(2022)` para cada bloco `b`.
3. **Regressão**: o swing é explicado por mínimos quadrados ponderados pelos votos,
   com covariáveis de *perfil* (share do bloco em 2022) e *porte* (log do eleitorado
   do município).
4. **Especialização progressiva**: a regressão é hierárquica com encolhimento (ridge)
   em direção ao nível acima: UF → região → nacional → pesquisa. Os pesos
   (`--peso-prior`, `--lambda-regiao`, `--lambda-uf`, em votos-equivalentes)
   controlam quantos votos apurados são necessários para que cada nível "ganhe" do
   nível acima. Com 1% das urnas vale a regra nacional; com 5% já há regra por UF.
5. **Projeção**: para cada local pendente, `share_proj = share_2022 + swing_previsto`,
   e os votos válidos esperados são os de 2022 (ajustados pelo nº de seções atual)
   vezes a razão de comparecimento observada nos locais já apurados (também
   hierárquica). O resultado projetado é `votos apurados + votos projetados`.

**Segundo turno.** Como o autor explica na entrevista, no 2º turno o modelo usa duas
referências ao mesmo tempo: o 2º turno de 2022 (base do swing) e o 1º turno de 2026,
cujo swing já observado em relação a 2022 entra como covariável adicional da regressão
(`--referencia 2022:2,2026:1`).

Para comparação, a simulação também calcula a **parcial ingênua** (o que a TV mostra:
fração dos votos já contados) e a **parcial ponderada por município**, hipótese que o
autor testou e descartou porque as primeiras urnas de um município não são amostra dele.

## Ordem de apuração (`apuracao/ordem.py`)

O arquivo de votação por seção do TSE não traz o horário em que cada urna foi
totalizada. Esse horário está nos arquivos de **boletim de urna** ("bweb", um zip por
UF, 5,1 GB no total para o 1º turno de 2026), em `resources/bweb_2026_1t/`.
`apuracao/bweb.py` extrai de cada zip, em streaming, uma linha por seção com o campo
`DT_BU_RECEBIDO` e grava `resultados/ordem_bweb_2026_1t.csv`: a ordem real de cada local
de votação (mediana do horário de recebimento das suas seções). Todas as 499.206 seções do
1º turno têm horário (17h08 de 4/10 a 2h58 de 5/10; 50% recebidas até 18h44).

Modos de `--ordem`:

- `auto` (padrão): a ordem real dos boletins de urna, se o CSV acima existir; senão
  `oficial-2026`.
- `oficial-2026` e `oficial-2022`: um deslocamento por UF e um peso para o porte do
  município, calibrados por `apuracao/calibrar_ordem.py` para que a totalização parcial
  simulada reproduza a curva oficial "% do candidato × % de urnas apuradas" do 2º turno.
  A de 2026 é calibrada na série oficial versão a versão (abaixo) com erro de 0,09 p.p.;
  a de 2022, em 14 pontos lidos do infográfico do g1.
- `realista`: viés estilizado (municípios pequenos e Sul/Sudeste/Centro-Oeste antes das
  capitais e do Nordeste), parecido com 2022.
- `aleatoria`: controle, amostra representativa.
- um CSV `uf,cd_municipio,nr_local,ordem` qualquer.

`resources/apuracao-brasil/` tem a **série oficial do 2º turno** (369 versões do boletim
nacional do TSE, de 17h06 a 23h27 de 25/10, coletadas pelo projeto Contagem Direta).
`apuracao/comparar_ordem.py` mede quanto cada ordem reproduz essa série:

| ordem | erro RMS da parcial de Lula vs. série oficial |
|---|---:|
| `oficial-2026` (calibrada) | 0,09 p.p. |
| `aleatoria` | 0,95 p.p. |
| ordem real dos BUs do 1º turno | 2,03 p.p. |
| `realista` (padrão 2022) | 3,17 p.p. |

A ordem real do 1º turno não serve de aproximação para o 2º: no 1º turno a apuração foi
fortemente enviesada (Lula com cerca de 40% da parcial entre 5% e 30% das urnas, contra
45,2% no final), enquanto no 2º turno a parcial nunca ficou abaixo de 49,5% para Lula,
mais perto de uma amostra aleatória. Os boletins de urna do 2º turno ainda não foram
publicados pelo TSE; quando forem, basta rodar `apuracao.bweb` apontando para a pasta e
passar o CSV em `--ordem`.

## Uso

```bash
uv run apuracao --turno 1 --prior "lula=46,bolsonaro=47"
```

```bash
uv run apuracao --turno 2 --referencia 2022:2,2026:1 --prior "lula=50,bolsonaro=50" --seed 7
```

```bash
uv run apuracao --turno 2 --referencia 2026:1 --ordem aleatoria
```

A primeira execução converte os CSVs (1,5 GB cada) em parquet em `.cache/` e demora
cerca de um minuto; as seguintes levam segundos. A tabela de cada simulação é salva em
`resultados/`.

## Resultados

1º turno de 2026, **ordem real dos boletins de urna**, referência 2022, prior 47/46:

| % seções apuradas | parcial Lula | parcial Flávio | projeção Lula | projeção Flávio |
|---:|---:|---:|---:|---:|
| 1,0% | 45,1% | 47,1% | 45,54% | 47,25% |
| 2,2% | 42,1% | 49,8% | 45,41% | 47,17% |
| 5,0% | 40,8% | 50,8% | 45,47% | 46,81% |
| 10% | 40,4% | 51,3% | 45,16% | 46,97% |
| 20% | 40,4% | 51,5% | 45,23% | 46,90% |
| 50% | 41,8% | 50,2% | 45,09% | 47,07% |
| final | 45,16% | 47,03% | | |

Com 2,2% das urnas (o ponto em que o autor publicou sua projeção) o erro é de 0,25 p.p.
para Lula e 0,14 p.p. para Flávio; a matéria relata 0,34 e 0,02 p.p. para o modelo
original. A partir de 10% das urnas o erro fica abaixo de 0,1 p.p., enquanto a
totalização parcial ainda mostrava Flávio 11 pontos à frente.

2º turno de 2026, ordem `oficial-2026`, referências 2022 (2º turno) + 2026 (1º turno),
prior 50/50. A coluna "parcial" é a série oficial do TSE:

| % seções apuradas | parcial Lula | projeção Lula | parcial Flávio | projeção Flávio |
|---:|---:|---:|---:|---:|
| 2,0% | 51,8% | 51,45% | 48,2% | 48,55% |
| 10% | 50,5% | 51,80% | 49,5% | 48,20% |
| 20% | 49,6% | 51,81% | 50,4% | 48,19% |
| 30% | 49,5% | 51,80% | 50,5% | 48,20% |
| 50% | 50,1% | 51,79% | 49,9% | 48,21% |
| final | 51,79% | | 48,21% | |

Enquanto a totalização oficial mostrava Flávio na frente entre 20% e 40% das urnas, a
projeção apontava Lula com 51,8% desde 10% apuradas. Esse número depende da ordem
simulada: é o resultado do modelo numa apuração que segue a curva oficial, não uma
reprodução da noite real urna a urna.

## Dashboard interativo

```bash
uv run python -m apuracao.exportar
```

```bash
uv run python -m apuracao.dashboard
```

O primeiro comando roda a simulação dos dois turnos a cada 1% das seções e salva
`resultados/dashboard.json` (1º turno com a ordem real dos BUs; 2º turno com a ordem
calibrada e a série oficial do TSE anexada); o segundo monta `resultados/dashboard.html`
(template em `apuracao/dashboard_template.html`), uma página com um slider de % de urnas
apuradas que mostra a totalização até aquele instante (no 2º turno, o boletim oficial do
TSE com seu horário), a projeção do modelo e um gráfico da evolução das duas curvas
contra o resultado final.

`apuracao.dashboard` também grava `site/index.html`, a mesma página embrulhada em um
documento HTML completo, que é o que o Vercel publica.

## Publicação no Vercel

O site é estático (um único `index.html` com os dados embutidos), então não há build nem
servidor: `vercel.json` aponta para a pasta `site/` e `.vercelignore` deixa de fora os
dados e o código Python. Basta importar o repositório no painel do Vercel (ou rodar
`vercel --prod` na pasta do projeto); cada push na `main` republica o site. Para
atualizar os números: rodar `apuracao.exportar` e `apuracao.dashboard` localmente,
commitar `site/index.html` e dar push.

Outros utilitários: `apuracao.bweb` (extrai a ordem real dos boletins de urna),
`apuracao.calibrar_ordem` (calibra uma ordem simulada numa curva oficial) e
`apuracao.comparar_ordem` (compara ordens com a série oficial do 2º turno).
