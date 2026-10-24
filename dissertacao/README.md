# Dissertação — MPPG / FGV

Projeto acadêmico: governança dos riscos decisórios na Perícia Médica Federal.

## Organização

- `00_gestao/`: orientações, planejamento e requerimentos LAI.
- `01_projeto/minuta_ativa/`: **única** minuta corrente (TOML + `conteudo.org`).
- `01_projeto/protocolo_revisao/`: protocolo de revisão de escopo.
- `02_referencias/`: normas, guias, bibliografia e referencial teórico.
- `03_dados/`: dados administrativos originais/tratados; **não versionados**.
- `04_analises/`: código, notas analíticas, modelos e decisões metodológicas.
- `05_redacao/`: capítulos e modelos da futura dissertação.
- `06_entregas/`: cópias de entregas finais (não versionadas).
- `90_historico/`: pacotes recebidos e projetos anteriores (não versionados).

## Política de versões

A pasta `minuta_ativa` mantém sempre o mesmo nome. Alterações são registradas
por commits Git e, quando importante, tags (`minuta-v1`, `minuta-v2`, etc.).
Não se criam `minuta_ativa_2`, `minuta_final_final` ou cópias do projeto a cada
edição. A designação `minuta_original` deve ficar reservada ao documento
original efetivamente identificado, não à primeira versão encontrada.

## Compilação

Do diretório do software MPPG:

```sh
PROJETO="$HOME/Documentos/mppg/dissertacao/01_projeto/minuta_ativa"
pipenv run python -m academic_pipeline minuta build \
  --config "$PROJETO/minuta.toml" --overwrite
```

Recompilar um Org específico no diretório do software:

```sh
pipenv run python -m academic_pipeline --recompile --org \
  "$HOME/Documentos/mppg/dissertacao/01_projeto/minuta_ativa/saida/minuta_pmf_buta/minuta_pmf_buta.org"
```

## Privacidade

Não publique este repositório sem revisar dados pessoais, documentos LAI,
conteúdo dos commits e arquivos já rastreados. O `.gitignore` não remove
arquivos que já tenham sido adicionados ao Git.
