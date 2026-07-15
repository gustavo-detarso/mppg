# AP-003B — preparação da extração do parser

> Relatório somente leitura. Nenhum módulo produtivo foi alterado.

- Arquivo: `app_bundle/scripts/pipeline/academic_pipeline_rc10.py`
- Primeiro `main()`: linhas 899–1682
- Nomes identificados para o parser: `['parser']`
- Operações relacionadas ao parser: **64**

## Operações detectadas

```json
[
  {
    "line": 900,
    "end_line": 900,
    "callable": "argparse.ArgumentParser",
    "source": "argparse.ArgumentParser(description=f\"academic_pipeline {PIPELINE_VERSION} — document_model canônico\")"
  },
  {
    "line": 901,
    "end_line": 901,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--config\", default=\"\", help=\"Arquivo TOML\")"
  },
  {
    "line": 902,
    "end_line": 902,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--tui\", action=\"store_true\", help=\"Abre a Central Operacional FGV em terminal (prompt_toolkit)\")"
  },
  {
    "line": 903,
    "end_line": 903,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--gui\", action=\"store_true\", help=\"Abre a interface gráfica FGV de atividades acadêmicas\")"
  },
  {
    "line": 904,
    "end_line": 904,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--init-toml\", action=\"store_true\", help=\"Abre o gerador interativo completo de TOML\")"
  },
  {
    "line": 905,
    "end_line": 905,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--toml-profile\", default=\"\", help=\"Preset inicial para --init-toml, ex.: atividade_local_fgv\")"
  },
  {
    "line": 906,
    "end_line": 906,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--no-clear\", action=\"store_true\", help=\"Não limpa a tela entre etapas do --init-toml\")"
  },
  {
    "line": 907,
    "end_line": 907,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--list-toml-profiles\", action=\"store_true\", help=\"Lista presets do gerador de TOML\")"
  },
  {
    "line": 908,
    "end_line": 908,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--list-institutions\", action=\"store_true\", help=\"Lista perfis institucionais disponíveis\")"
  },
  {
    "line": 909,
    "end_line": 909,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--list-layouts\", action=\"store_true\", help=\"Lista layouts disponíveis do perfil institucional informado no TOML\")"
  },
  {
    "line": 910,
    "end_line": 910,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--explain-profile\", default=\"\", nargs=\"?\", const=\"fgv\", help=\"Explica um perfil institucional, ex.: --explain-profile fgv\")"
  },
  {
    "line": 911,
    "end_line": 911,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--show-prompts\", action=\"store_true\", help=\"Mostra os prompts/diretivas ativos para o TOML informado\")"
  },
  {
    "line": 912,
    "end_line": 912,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--write-prompt-lock\", action=\"store_true\", help=\"Gera prompt_lock.json/md para o TOML e encerra\")"
  },
  {
    "line": 913,
    "end_line": 913,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--check-institution-compliance\", action=\"store_true\", help=\"Valida conformidade institucional de artefatos já gerados\")"
  },
  {
    "line": 914,
    "end_line": 914,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--doctor\", action=\"store_true\", help=\"Diagnostica ambiente, ferramentas, arquivos FGV e estilo bibliográfico\")"
  },
  {
    "line": 915,
    "end_line": 915,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--check-config\", action=\"store_true\", help=\"Valida preventivamente o TOML e encerra\")"
  },
  {
    "line": 916,
    "end_line": 916,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--recompile\", action=\"store_true\", help=\"Recompila um .org existente sem chamar IA\")"
  },
  {
    "line": 917,
    "end_line": 917,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--org\", default=\"\", help=\"Arquivo .org para --recompile\")"
  },
  {
    "line": 918,
    "end_line": 918,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--academic-writing\", default=\"\", help=\"Override do academic-writing.el para --recompile\")"
  },
  {
    "line": 919,
    "end_line": 919,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--latex-extra-path\", default=\"\", help=\"Override do latex_extra_path para --recompile\")"
  },
  {
    "line": 920,
    "end_line": 920,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--pdf-engine\", default=\"\", help=\"Override do pdf_engine para --recompile\")"
  },
  {
    "line": 921,
    "end_line": 921,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--no-clean\", action=\"store_true\", help=\"Não remove auxiliares no --recompile\")"
  },
  {
    "line": 922,
    "end_line": 922,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--somente-renderizar\", action=\"store_true\", help=\"Usa document.json existente e só renderiza saídas\")"
  },
  {
    "line": 923,
    "end_line": 923,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--somente-mapa-mental\", action=\"store_true\", help=\"Usa document.json existente e gera/renderiza apenas o mapa mental\")"
  },
  {
    "line": 924,
    "end_line": 924,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--reusar-mapa-mental\", action=\"store_true\", help=\"Reaproveita imagem de mapa mental existente quando disponível, sem chamar IA/PlantUML\")"
  },
  {
    "line": 925,
    "end_line": 925,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--forcar-regeneracao-mapa-mental\", action=\"store_true\", help=\"Remove mapa mental existente e recria PlantUML/imagem quando a etapa de mapa mental for executada\")"
  },
  {
    "line": 926,
    "end_line": 926,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--document-json\", default=\"\", help=\"Caminho de document.json existente\")"
  },
  {
    "line": 927,
    "end_line": 927,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--prisma-importar-triagem\", default=\"\", help=\"Importa CSV de triagem humana do perfil relatorio_prisma_busca_orientada_fgv e consolida matriz/relatório PRISMA\")"
  },
  {
    "line": 928,
    "end_line": 928,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--init-project\", default=\"\", help=\"Cria app_bundle/projetos/<nome> com TOML, ZIPs placeholder e doi_manifest.csv\")"
  },
  {
    "line": 929,
    "end_line": 929,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--project-type\", default=\"paper\", choices=[\"paper\", \"atividade\", \"prisma\", \"atividade_prisma\", \"paper_prisma\"], help=\"Tipo de projeto para --init-project\")"
  },
  {
    "line": 930,
    "end_line": 930,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--institution\", default=\"fgv\", help=\"Perfil institucional usado por --init-project, ex.: fgv\")"
  },
  {
    "line": 931,
    "end_line": 931,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--base-dir\", default=\"\", help=\"Raiz do academic_pipeline ou app_bundle para --init-project\")"
  },
  {
    "line": 932,
    "end_line": 932,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--overwrite-project\", action=\"store_true\", help=\"Permite sobrescrever arquivos seguros criados por --init-project\")"
  },
  {
    "line": 933,
    "end_line": 933,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--make-doi-manifest\", action=\"store_true\", help=\"Gera doi_manifest.csv a partir de --input-zip ou --input-dir\")"
  },
  {
    "line": 934,
    "end_line": 934,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--input-zip\", default=\"\", help=\"ZIP de documentos para --make-doi-manifest\")"
  },
  {
    "line": 935,
    "end_line": 935,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--input-dir\", default=\"\", help=\"Pasta de documentos para --make-doi-manifest\")"
  },
  {
    "line": 936,
    "end_line": 936,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--output\", default=\"\", help=\"Arquivo de saída para --make-doi-manifest\")"
  },
  {
    "line": 937,
    "end_line": 937,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--output-dir\", default=\"\", help=\"Override de [paths].document_output_dir para a geração/renderização do documento\")"
  },
  {
    "line": 938,
    "end_line": 938,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--work-dir\", default=\"\", help=\"Override de [paths].work_dir para extrações temporárias\")"
  },
  {
    "line": 939,
    "end_line": 939,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--cache-dir\", default=\"\", help=\"Override de [paths].cache_dir para fulltext_cache\")"
  },
  {
    "line": 940,
    "end_line": 940,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--research-output-dir\", default=\"\", help=\"Override de [paths].research_output_dir para relatório PRISMA\")"
  },
  {
    "line": 941,
    "end_line": 941,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--output-prefix\", default=\"\", help=\"Override de [paths].document_prefix\")"
  },
  {
    "line": 942,
    "end_line": 942,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--layout\", default=\"\", help=\"Override de [documento].layout, ex.: atividade_fgv\")"
  },
  {
    "line": 943,
    "end_line": 943,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--tipo-conteudo\", default=\"\", help=\"Override de [documento].tipo_conteudo, ex.: resumo_artigos\")"
  },
  {
    "line": 944,
    "end_line": 944,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--genero-academico\", default=\"\", help=\"Override de [documento].genero_academico, ex.: atividade\")"
  },
  {
    "line": 945,
    "end_line": 945,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--no-output-subdir\", action=\"store_true\", help=\"Não cria subdiretório com document_prefix dentro de --output-dir/[paths].document_output_dir\")"
  },
  {
    "line": 946,
    "end_line": 946,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--inspect-bib\", default=\"\", help=\"Inspeciona arquivo .bib e gera relatório .md/.json\")"
  },
  {
    "line": 947,
    "end_line": 947,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--quality-report\", action=\"store_true\", help=\"Gera quality_report.md a partir de --document-json e opcionalmente --org\")"
  },
  {
    "line": 948,
    "end_line": 948,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--bib\", default=\"\", help=\"Arquivo .bib opcional para --quality-report ou --check-institution-compliance\")"
  },
  {
    "line": 949,
    "end_line": 949,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--docx\", default=\"\", help=\"Arquivo .docx opcional para --check-institution-compliance\")"
  },
  {
    "line": 950,
    "end_line": 950,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\"--pdf\", default=\"\", help=\"Arquivo .pdf opcional para --check-institution-compliance\")"
  },
  {
    "line": 952,
    "end_line": 956,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-menu\",\n        action=\"store_true\",\n        help=\"Abre o sub-menu PRISMA de curadoria IA de referências.\",\n    )"
  },
  {
    "line": 957,
    "end_line": 961,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-ia\",\n        action=\"store_true\",\n        help=\"Executa a curadoria IA v2 de referências e gera XLSX/CSV para o PRISMA.\",\n    )"
  },
  {
    "line": 962,
    "end_line": 966,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-sem-ia\",\n        action=\"store_true\",\n        help=\"Usa a etapa de curadoria sem chamada à IA, apenas com heurística local.\",\n    )"
  },
  {
    "line": 967,
    "end_line": 971,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-reexportar-xlsx\",\n        action=\"store_true\",\n        help=\"Reexporta o XLSX de curadoria revisado para triagem_humana.csv.\",\n    )"
  },
  {
    "line": 972,
    "end_line": 976,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-importar\",\n        action=\"store_true\",\n        help=\"Importa triagem_humana.csv e executa a geração final do PRISMA.\",\n    )"
  },
  {
    "line": 977,
    "end_line": 981,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-fluxo-completo\",\n        action=\"store_true\",\n        help=\"Executa curadoria IA e depois importa a triagem para gerar o PRISMA final.\",\n    )"
  },
  {
    "line": 982,
    "end_line": 986,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-prompt\",\n        default=\"\",\n        help=\"Caminho do YAML de prompt estruturado da curadoria.\",\n    )"
  },
  {
    "line": 987,
    "end_line": 991,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-input\",\n        default=\"\",\n        help=\"Entrada específica para a curadoria: XLSX/CSV de triagem ou XLSX revisado.\",\n    )"
  },
  {
    "line": 992,
    "end_line": 996,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-out-dir\",\n        default=\"\",\n        help=\"Diretório de saída do relatório PRISMA.\",\n    )"
  },
  {
    "line": 997,
    "end_line": 1002,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-max-incluir\",\n        type=int,\n        default=0,\n        help=\"Número máximo de referências incluídas pela curadoria.\",\n    )"
  },
  {
    "line": 1003,
    "end_line": 1008,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-top-n-candidatos\",\n        type=int,\n        default=0,\n        help=\"Número de candidatos enviados/avaliados pela curadoria IA.\",\n    )"
  },
  {
    "line": 1009,
    "end_line": 1014,
    "callable": "parser.add_argument",
    "source": "parser.add_argument(\n        \"--prisma-curadoria-limiar-minimo\",\n        type=int,\n        default=0,\n        help=\"Limiar mínimo de inclusão da curadoria v2.\",\n    )"
  },
  {
    "line": 1016,
    "end_line": 1016,
    "callable": "parser.parse_args",
    "source": "parser.parse_args()"
  }
]
```

## Blocos-fonte candidatos à extração

### Bloco 1: linhas 900–950

```python
    897: 
    898: 
    899: def main() -> int:
>>  900:     parser = argparse.ArgumentParser(description=f"academic_pipeline {PIPELINE_VERSION} — document_model canônico")
>>  901:     parser.add_argument("--config", default="", help="Arquivo TOML")
>>  902:     parser.add_argument("--tui", action="store_true", help="Abre a Central Operacional FGV em terminal (prompt_toolkit)")
>>  903:     parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica FGV de atividades acadêmicas")
>>  904:     parser.add_argument("--init-toml", action="store_true", help="Abre o gerador interativo completo de TOML")
>>  905:     parser.add_argument("--toml-profile", default="", help="Preset inicial para --init-toml, ex.: atividade_local_fgv")
>>  906:     parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre etapas do --init-toml")
>>  907:     parser.add_argument("--list-toml-profiles", action="store_true", help="Lista presets do gerador de TOML")
>>  908:     parser.add_argument("--list-institutions", action="store_true", help="Lista perfis institucionais disponíveis")
>>  909:     parser.add_argument("--list-layouts", action="store_true", help="Lista layouts disponíveis do perfil institucional informado no TOML")
>>  910:     parser.add_argument("--explain-profile", default="", nargs="?", const="fgv", help="Explica um perfil institucional, ex.: --explain-profile fgv")
>>  911:     parser.add_argument("--show-prompts", action="store_true", help="Mostra os prompts/diretivas ativos para o TOML informado")
>>  912:     parser.add_argument("--write-prompt-lock", action="store_true", help="Gera prompt_lock.json/md para o TOML e encerra")
>>  913:     parser.add_argument("--check-institution-compliance", action="store_true", help="Valida conformidade institucional de artefatos já gerados")
>>  914:     parser.add_argument("--doctor", action="store_true", help="Diagnostica ambiente, ferramentas, arquivos FGV e estilo bibliográfico")
>>  915:     parser.add_argument("--check-config", action="store_true", help="Valida preventivamente o TOML e encerra")
>>  916:     parser.add_argument("--recompile", action="store_true", help="Recompila um .org existente sem chamar IA")
>>  917:     parser.add_argument("--org", default="", help="Arquivo .org para --recompile")
>>  918:     parser.add_argument("--academic-writing", default="", help="Override do academic-writing.el para --recompile")
>>  919:     parser.add_argument("--latex-extra-path", default="", help="Override do latex_extra_path para --recompile")
>>  920:     parser.add_argument("--pdf-engine", default="", help="Override do pdf_engine para --recompile")
>>  921:     parser.add_argument("--no-clean", action="store_true", help="Não remove auxiliares no --recompile")
>>  922:     parser.add_argument("--somente-renderizar", action="store_true", help="Usa document.json existente e só renderiza saídas")
>>  923:     parser.add_argument("--somente-mapa-mental", action="store_true", help="Usa document.json existente e gera/renderiza apenas o mapa mental")
>>  924:     parser.add_argument("--reusar-mapa-mental", action="store_true", help="Reaproveita imagem de mapa mental existente quando disponível, sem chamar IA/PlantUML")
>>  925:     parser.add_argument("--forcar-regeneracao-mapa-mental", action="store_true", help="Remove mapa mental existente e recria PlantUML/imagem quando a etapa de mapa mental for executada")
>>  926:     parser.add_argument("--document-json", default="", help="Caminho de document.json existente")
>>  927:     parser.add_argument("--prisma-importar-triagem", default="", help="Importa CSV de triagem humana do perfil relatorio_prisma_busca_orientada_fgv e consolida matriz/relatório PRISMA")
>>  928:     parser.add_argument("--init-project", default="", help="Cria app_bundle/projetos/<nome> com TOML, ZIPs placeholder e doi_manifest.csv")
>>  929:     parser.add_argument("--project-type", default="paper", choices=["paper", "atividade", "prisma", "atividade_prisma", "paper_prisma"], help="Tipo de projeto para --init-project")
>>  930:     parser.add_argument("--institution", default="fgv", help="Perfil institucional usado por --init-project, ex.: fgv")
>>  931:     parser.add_argument("--base-dir", default="", help="Raiz do academic_pipeline ou app_bundle para --init-project")
>>  932:     parser.add_argument("--overwrite-project", action="store_true", help="Permite sobrescrever arquivos seguros criados por --init-project")
>>  933:     parser.add_argument("--make-doi-manifest", action="store_true", help="Gera doi_manifest.csv a partir de --input-zip ou --input-dir")
>>  934:     parser.add_argument("--input-zip", default="", help="ZIP de documentos para --make-doi-manifest")
>>  935:     parser.add_argument("--input-dir", default="", help="Pasta de documentos para --make-doi-manifest")
>>  936:     parser.add_argument("--output", default="", help="Arquivo de saída para --make-doi-manifest")
>>  937:     parser.add_argument("--output-dir", default="", help="Override de [paths].document_output_dir para a geração/renderização do documento")
>>  938:     parser.add_argument("--work-dir", default="", help="Override de [paths].work_dir para extrações temporárias")
>>  939:     parser.add_argument("--cache-dir", default="", help="Override de [paths].cache_dir para fulltext_cache")
>>  940:     parser.add_argument("--research-output-dir", default="", help="Override de [paths].research_output_dir para relatório PRISMA")
>>  941:     parser.add_argument("--output-prefix", default="", help="Override de [paths].document_prefix")
>>  942:     parser.add_argument("--layout", default="", help="Override de [documento].layout, ex.: atividade_fgv")
>>  943:     parser.add_argument("--tipo-conteudo", default="", help="Override de [documento].tipo_conteudo, ex.: resumo_artigos")
>>  944:     parser.add_argument("--genero-academico", default="", help="Override de [documento].genero_academico, ex.: atividade")
>>  945:     parser.add_argument("--no-output-subdir", action="store_true", help="Não cria subdiretório com document_prefix dentro de --output-dir/[paths].document_output_dir")
>>  946:     parser.add_argument("--inspect-bib", default="", help="Inspeciona arquivo .bib e gera relatório .md/.json")
>>  947:     parser.add_argument("--quality-report", action="store_true", help="Gera quality_report.md a partir de --document-json e opcionalmente --org")
>>  948:     parser.add_argument("--bib", default="", help="Arquivo .bib opcional para --quality-report ou --check-institution-compliance")
>>  949:     parser.add_argument("--docx", default="", help="Arquivo .docx opcional para --check-institution-compliance")
>>  950:     parser.add_argument("--pdf", default="", help="Arquivo .pdf opcional para --check-institution-compliance")
    951:     # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS >>>
    952:     parser.add_argument(
    953:         "--prisma-curadoria-menu",
```

### Bloco 2: linhas 952–1014

```python
    949:     parser.add_argument("--docx", default="", help="Arquivo .docx opcional para --check-institution-compliance")
    950:     parser.add_argument("--pdf", default="", help="Arquivo .pdf opcional para --check-institution-compliance")
    951:     # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS >>>
>>  952:     parser.add_argument(
>>  953:         "--prisma-curadoria-menu",
>>  954:         action="store_true",
>>  955:         help="Abre o sub-menu PRISMA de curadoria IA de referências.",
>>  956:     )
>>  957:     parser.add_argument(
>>  958:         "--prisma-curadoria-ia",
>>  959:         action="store_true",
>>  960:         help="Executa a curadoria IA v2 de referências e gera XLSX/CSV para o PRISMA.",
>>  961:     )
>>  962:     parser.add_argument(
>>  963:         "--prisma-curadoria-sem-ia",
>>  964:         action="store_true",
>>  965:         help="Usa a etapa de curadoria sem chamada à IA, apenas com heurística local.",
>>  966:     )
>>  967:     parser.add_argument(
>>  968:         "--prisma-curadoria-reexportar-xlsx",
>>  969:         action="store_true",
>>  970:         help="Reexporta o XLSX de curadoria revisado para triagem_humana.csv.",
>>  971:     )
>>  972:     parser.add_argument(
>>  973:         "--prisma-curadoria-importar",
>>  974:         action="store_true",
>>  975:         help="Importa triagem_humana.csv e executa a geração final do PRISMA.",
>>  976:     )
>>  977:     parser.add_argument(
>>  978:         "--prisma-curadoria-fluxo-completo",
>>  979:         action="store_true",
>>  980:         help="Executa curadoria IA e depois importa a triagem para gerar o PRISMA final.",
>>  981:     )
>>  982:     parser.add_argument(
>>  983:         "--prisma-curadoria-prompt",
>>  984:         default="",
>>  985:         help="Caminho do YAML de prompt estruturado da curadoria.",
>>  986:     )
>>  987:     parser.add_argument(
>>  988:         "--prisma-curadoria-input",
>>  989:         default="",
>>  990:         help="Entrada específica para a curadoria: XLSX/CSV de triagem ou XLSX revisado.",
>>  991:     )
>>  992:     parser.add_argument(
>>  993:         "--prisma-curadoria-out-dir",
>>  994:         default="",
>>  995:         help="Diretório de saída do relatório PRISMA.",
>>  996:     )
>>  997:     parser.add_argument(
>>  998:         "--prisma-curadoria-max-incluir",
>>  999:         type=int,
>> 1000:         default=0,
>> 1001:         help="Número máximo de referências incluídas pela curadoria.",
>> 1002:     )
>> 1003:     parser.add_argument(
>> 1004:         "--prisma-curadoria-top-n-candidatos",
>> 1005:         type=int,
>> 1006:         default=0,
>> 1007:         help="Número de candidatos enviados/avaliados pela curadoria IA.",
>> 1008:     )
>> 1009:     parser.add_argument(
>> 1010:         "--prisma-curadoria-limiar-minimo",
>> 1011:         type=int,
>> 1012:         default=0,
>> 1013:         help="Limiar mínimo de inclusão da curadoria v2.",
>> 1014:     )
   1015:     # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS <<<
   1016:     args = parser.parse_args()
   1017: 
```

### Bloco 3: linhas 1016–1016

```python
   1013:         help="Limiar mínimo de inclusão da curadoria v2.",
   1014:     )
   1015:     # <<< PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_ARGS <<<
>> 1016:     args = parser.parse_args()
   1017: 
   1018:     # >>> PATCH_PRISMA_CURADORIA_IA_MENU_PRINCIPAL_V1_DISPATCH >>>
   1019:     if (
```
