# Bundle profissional — pesquisa e documento acadêmico

Este bundle organiza o projeto em uma estrutura interna mais próxima de um software profissional.

## Componentes principais

- motor de pesquisa: `scripts/research/gerador_pesquisa_rc_2.py`
- pipeline integrado: `scripts/pipeline/gerador_pesquisa_documento_rc_2.py`
- motor standalone de documento acadêmico: `scripts/document/gerador_documento_academico_rc_3.py`
- núcleo de geração textual: `scripts/document/gerar_documento_org_ai_interativo_rc_1.py`

## Tipos documentais previstos

No motor standalone e no pipeline, o documento final pode ser:
- `paper`
- `dissertacao`

## Importante

Execute os comandos a partir da **raiz do bundle** para que os caminhos relativos dos TOMLs funcionem corretamente.

## Quickstart

### Só pesquisa
```bash
python ./scripts/research/gerador_pesquisa_rc_2.py --config ./config/research/template_toml_unificado_rc_2.toml
```

### Pipeline completo
```bash
python ./scripts/pipeline/gerador_pesquisa_documento_rc_2.py --config ./config/pipeline/template_toml_pipeline_pesquisa_documento_rc_2.toml
```

### Só documento acadêmico
```bash
python ./scripts/document/gerador_documento_academico_rc_3.py --config ./config/document/template_toml_documento_academico_rc_3.toml
```

Leia também:
- `docs/manual_unificado_rc_18.md`

## Diagrama

- `docs/diagrama_arquitetura.md`
- `docs/diagrama_arquitetura.svg`
