from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser(*, pipeline_version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"academic_pipeline {pipeline_version} — document_model canônico")
    parser.add_argument("--config", default="", help="Arquivo TOML")
    parser.add_argument("--tui", action="store_true", help="Abre a Central Operacional FGV em terminal (prompt_toolkit)")
    parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica FGV de atividades acadêmicas")
    parser.add_argument("--init-toml", action="store_true", help="Abre o gerador interativo completo de TOML")
    parser.add_argument("--toml-profile", default="", help="Preset inicial para --init-toml, ex.: atividade_local_fgv")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre etapas do --init-toml")
    parser.add_argument("--list-toml-profiles", action="store_true", help="Lista presets do gerador de TOML")
    parser.add_argument("--list-institutions", action="store_true", help="Lista perfis institucionais disponíveis")
    parser.add_argument("--list-layouts", action="store_true", help="Lista layouts disponíveis do perfil institucional informado no TOML")
    parser.add_argument("--explain-profile", default="", nargs="?", const="fgv", help="Explica um perfil institucional, ex.: --explain-profile fgv")
    parser.add_argument("--show-prompts", action="store_true", help="Mostra os prompts/diretivas ativos para o TOML informado")
    parser.add_argument("--write-prompt-lock", action="store_true", help="Gera prompt_lock.json/md para o TOML e encerra")
    parser.add_argument("--check-institution-compliance", action="store_true", help="Valida conformidade institucional de artefatos já gerados")
    parser.add_argument("--doctor", action="store_true", help="Diagnostica ambiente, ferramentas, arquivos FGV e estilo bibliográfico")
    parser.add_argument("--check-config", action="store_true", help="Valida preventivamente o TOML e encerra")
    parser.add_argument("--recompile", action="store_true", help="Recompila um .org existente sem chamar IA")
    parser.add_argument("--org", default="", help="Arquivo .org para --recompile")
    parser.add_argument("--academic-writing", default="", help="Override do academic-writing.el para --recompile")
    parser.add_argument("--latex-extra-path", default="", help="Override do latex_extra_path para --recompile")
    parser.add_argument("--pdf-engine", default="", help="Override do pdf_engine para --recompile")
    parser.add_argument("--no-clean", action="store_true", help="Não remove auxiliares no --recompile")
    parser.add_argument("--somente-renderizar", action="store_true", help="Usa document.json existente e só renderiza saídas")
    parser.add_argument("--somente-mapa-mental", action="store_true", help="Usa document.json existente e gera/renderiza apenas o mapa mental")
    parser.add_argument("--reusar-mapa-mental", action="store_true", help="Reaproveita imagem de mapa mental existente quando disponível, sem chamar IA/PlantUML")
    parser.add_argument("--forcar-regeneracao-mapa-mental", action="store_true", help="Remove mapa mental existente e recria PlantUML/imagem quando a etapa de mapa mental for executada")
    parser.add_argument("--document-json", default="", help="Caminho de document.json existente")
    parser.add_argument("--prisma-importar-triagem", default="", help="Importa CSV de triagem humana do perfil relatorio_prisma_busca_orientada_fgv e consolida matriz/relatório PRISMA")
    parser.add_argument("--init-project", default="", help="Cria app_bundle/projetos/<nome> com TOML, ZIPs placeholder e doi_manifest.csv")
    parser.add_argument("--project-type", default="paper", choices=["paper", "atividade", "prisma", "atividade_prisma", "paper_prisma"], help="Tipo de projeto para --init-project")
    parser.add_argument("--institution", default="fgv", help="Perfil institucional usado por --init-project, ex.: fgv")
    parser.add_argument("--base-dir", default="", help="Raiz do academic_pipeline ou app_bundle para --init-project")
    parser.add_argument("--overwrite-project", action="store_true", help="Permite sobrescrever arquivos seguros criados por --init-project")
    parser.add_argument("--make-doi-manifest", action="store_true", help="Gera doi_manifest.csv a partir de --input-zip ou --input-dir")
    parser.add_argument("--input-zip", default="", help="ZIP de documentos para --make-doi-manifest")
    parser.add_argument("--input-dir", default="", help="Pasta de documentos para --make-doi-manifest")
    parser.add_argument("--output", default="", help="Arquivo de saída para --make-doi-manifest")
    parser.add_argument("--output-dir", default="", help="Override de [paths].document_output_dir para a geração/renderização do documento")
    parser.add_argument("--work-dir", default="", help="Override de [paths].work_dir para extrações temporárias")
    parser.add_argument("--cache-dir", default="", help="Override de [paths].cache_dir para fulltext_cache")
    parser.add_argument("--research-output-dir", default="", help="Override de [paths].research_output_dir para relatório PRISMA")
    parser.add_argument("--output-prefix", default="", help="Override de [paths].document_prefix")
    parser.add_argument("--layout", default="", help="Override de [documento].layout, ex.: atividade_fgv")
    parser.add_argument("--tipo-conteudo", default="", help="Override de [documento].tipo_conteudo, ex.: resumo_artigos")
    parser.add_argument("--genero-academico", default="", help="Override de [documento].genero_academico, ex.: atividade")
    parser.add_argument("--no-output-subdir", action="store_true", help="Não cria subdiretório com document_prefix dentro de --output-dir/[paths].document_output_dir")
    parser.add_argument("--inspect-bib", default="", help="Inspeciona arquivo .bib e gera relatório .md/.json")
    parser.add_argument("--quality-report", action="store_true", help="Gera quality_report.md a partir de --document-json e opcionalmente --org")
    parser.add_argument("--bib", default="", help="Arquivo .bib opcional para --quality-report ou --check-institution-compliance")
    parser.add_argument("--docx", default="", help="Arquivo .docx opcional para --check-institution-compliance")
    parser.add_argument("--pdf", default="", help="Arquivo .pdf opcional para --check-institution-compliance")
    parser.add_argument(
            "--prisma-curadoria-menu",
            action="store_true",
            help="Abre o sub-menu PRISMA de curadoria IA de referências.",
        )
    parser.add_argument(
            "--prisma-curadoria-ia",
            action="store_true",
            help="Executa a curadoria IA v2 de referências e gera XLSX/CSV para o PRISMA.",
        )
    parser.add_argument(
            "--prisma-curadoria-sem-ia",
            action="store_true",
            help="Usa a etapa de curadoria sem chamada à IA, apenas com heurística local.",
        )
    parser.add_argument(
            "--prisma-curadoria-reexportar-xlsx",
            action="store_true",
            help="Reexporta o XLSX de curadoria revisado para triagem_humana.csv.",
        )
    parser.add_argument(
            "--prisma-curadoria-importar",
            action="store_true",
            help="Importa triagem_humana.csv e executa a geração final do PRISMA.",
        )
    parser.add_argument(
            "--prisma-curadoria-fluxo-completo",
            action="store_true",
            help="Executa curadoria IA e depois importa a triagem para gerar o PRISMA final.",
        )
    parser.add_argument(
            "--prisma-curadoria-prompt",
            default="",
            help="Caminho do YAML de prompt estruturado da curadoria.",
        )
    parser.add_argument(
            "--prisma-curadoria-input",
            default="",
            help="Entrada específica para a curadoria: XLSX/CSV de triagem ou XLSX revisado.",
        )
    parser.add_argument(
            "--prisma-curadoria-out-dir",
            default="",
            help="Diretório de saída do relatório PRISMA.",
        )
    parser.add_argument(
            "--prisma-curadoria-max-incluir",
            type=int,
            default=0,
            help="Número máximo de referências incluídas pela curadoria.",
        )
    parser.add_argument(
            "--prisma-curadoria-top-n-candidatos",
            type=int,
            default=0,
            help="Número de candidatos enviados/avaliados pela curadoria IA.",
        )
    parser.add_argument(
            "--prisma-curadoria-limiar-minimo",
            type=int,
            default=0,
            help="Limiar mínimo de inclusão da curadoria v2.",
        )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    pipeline_version: str,
) -> argparse.Namespace:
    return build_parser(pipeline_version=pipeline_version).parse_args(argv)


__all__ = ["build_parser", "parse_args"]
