#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central operacional visual do Academic Pipeline rc10.7.42.

Esta TUI segue a convenção da Administração UAC: diálogos ``prompt_toolkit``
em tela inteira, lista vertical navegável, atalhos de teclado, confirmação
explícita antes de ações dispendiosas e campos de caminho com ``Tab`` para
conclusão automática. Não é uma GUI de desktop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Compatibilidade temporária entre pacote e script direto.
if __package__:
    from . import academic_pipeline_tui_widgets as ui
else:
    import academic_pipeline_tui_widgets as ui

try:
    from article_workflow import ArticleWorkflow
except Exception:  # pragma: no cover - fallback para instalações incompletas
    ArticleWorkflow = None  # type: ignore[assignment]


HERE = Path(__file__).resolve()
APP = HERE.parents[2]
ROOT = APP.parent
PIPELINE = HERE.with_name("pipeline_orchestrator.py")
TOML_WIZARD = HERE.with_name("academic_pipeline_toml_generator_interativo.py")
ARTICLE_FINALIZER = HERE.with_name("gerar_artigo_final_unificado.py")
STATE_PATH = APP / ".academic_pipeline_tui_state.json"
LOG_DIR = APP / "output" / "tui_logs"

# Perfis expostos no fluxo guiado. A TUI seleciona um deles antes de abrir
# o wizard; cada perfil mantém suas próprias perguntas específicas.
DOCUMENT_PROFILES: list[tuple[str, str]] = [
    ("atividade_local_fgv", "[1] Atividade acadêmica local FGV — exercício, resenha ou resposta de aula"),
    ("resumo_artigos_local_fgv", "[2] Resumo analítico de artigos FGV — análise, comparação e síntese"),
    ("paper_local_fgv", "[3] Paper acadêmico local FGV — paper a partir de corpus local"),
    ("paper_prisma_fgv", "[4] Paper + relatório PRISMA FGV — paper e trilha de pesquisa"),
    ("dissertacao_local_fgv", "[5] Dissertação local FGV — documento longo com elementos pré-textuais"),
    ("dissertacao_prisma_fgv", "[6] Dissertação + relatório PRISMA FGV — dissertação com pesquisa auditável"),
    ("relatorio_prisma_fgv", "[7] Relatório PRISMA autônomo FGV — somente pesquisa e triagem"),
    ("relatorio_prisma_busca_orientada_fgv", "[8] Relatório PRISMA com busca orientada FGV — busca externa, deduplicação e triagem humana"),
    ("somente_renderizar_fgv", "[9] Renderizar document.json existente FGV — sem nova chamada à IA"),
]



class AcademicPipelineTUI:
    """Coordena o fluxo configurar → conferir → validar → gerar → revisar."""

    def __init__(self, *, no_clear: bool = False) -> None:
        # Mantido por compatibilidade de CLI. A TUI prompt_toolkit controla a
        # própria tela, portanto não usa clear convencional.
        self.no_clear = bool(no_clear)
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # Estado e TOML
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if not STATE_PATH.exists():
            return {}
        try:
            value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def active_config(self) -> Path | None:
        raw = str(self.state.get("active_config") or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path.resolve() if path.exists() and path.is_file() else None

    def set_active_config(self, path: Path | None) -> None:
        if path is None:
            self.state.pop("active_config", None)
        else:
            resolved = str(path.resolve())
            self.state["active_config"] = resolved
            known = [str(x) for x in self.state.get("known_configs", []) if str(x).strip()]
            if resolved not in known:
                known.insert(0, resolved)
            self.state["known_configs"] = known[:30]
        self._save_state()

    @staticmethod
    def load_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def resolve_config_path(config_path: Path, raw: Any) -> Path | None:
        value = str(raw or "").strip()
        if not value or value.startswith("profile://"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return path.resolve()

    @classmethod
    def is_project_config(cls, path: Path) -> bool:
        try:
            cfg = cls.load_toml(path)
        except Exception:
            return False
        return bool(cls.section(cfg, "projeto")) and bool(cls.section(cfg, "documento"))

    @classmethod
    def is_activity_config(cls, path: Path) -> bool:
        """Compatibilidade com integrações antigas da rc10.7.39."""
        return cls.is_project_config(path)

    def list_project_configs(self) -> list[Path]:
        candidates: set[Path] = set()
        base = APP / "projetos"
        if base.exists():
            candidates.update(p.resolve() for p in base.rglob("*.toml") if p.is_file())
        for raw in self.state.get("known_configs", []):
            path = Path(str(raw)).expanduser()
            if path.exists() and path.is_file():
                candidates.add(path.resolve())
        active = self.active_config()
        if active:
            candidates.add(active)
        values = [p for p in candidates if self.is_project_config(p)]
        return sorted(values, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)

    def list_activity_configs(self) -> list[Path]:
        return self.list_project_configs()

    def cfg_output_dir(self, config_path: Path, cfg: dict[str, Any] | None = None) -> Path:
        cfg = cfg if cfg is not None else self.load_toml(config_path)
        paths = self.section(cfg, "paths")
        project = self.section(cfg, "projeto")
        raw_base = paths.get("document_output_dir") or "../../output/documento"
        base = self.resolve_config_path(config_path, raw_base) or (config_path.parent / "output" / "documento")
        prefix = str(paths.get("document_prefix") or project.get("nome") or "documento").strip() or "documento"
        return base / prefix if bool(paths.get("create_document_subdir", True)) else base

    def cfg_prefix(self, cfg: dict[str, Any]) -> str:
        paths = self.section(cfg, "paths")
        project = self.section(cfg, "projeto")
        return str(paths.get("document_prefix") or project.get("nome") or "documento").strip() or "documento"

    # ------------------------------------------------------------------
    # Diagnóstico visual
    # ------------------------------------------------------------------

    def input_rows(self, config_path: Path, cfg: dict[str, Any] | None = None) -> list[tuple[str, str, str]]:
        cfg = cfg if cfg is not None else self.load_toml(config_path)
        local = self.section(cfg, "documentos_locais")
        orientations = self.section(cfg, "orientacoes")
        research = self.section(cfg, "pesquisa")
        prompts = self.section(cfg, "prompts")
        rows: list[tuple[str, str, str]] = []

        def add_path(label: str, raw: Any, *, required: bool = False) -> None:
            value = str(raw or "").strip()
            if not value:
                rows.append((label, "—", "PENDENTE" if required else "não informado"))
                return
            if value.startswith("profile://"):
                rows.append((label, value, "interno do perfil"))
                return
            path = self.resolve_config_path(config_path, value)
            rows.append((label, str(path) if path else value, "OK" if path and path.exists() else "AUSENTE"))

        corpus_active = bool(local.get("ativos", True))
        input_zip = str(local.get("input_zip") or "").strip()
        input_dir = str(local.get("input_dir") or "").strip()
        if corpus_active:
            add_path("Corpus ZIP", input_zip, required=not input_dir)
            add_path("Corpus pasta", input_dir, required=not input_zip)
        else:
            rows.append(("Corpus local", "não exigido para este modo", "OK"))

        orient_paths = orientations.get("paths", [])
        if isinstance(orient_paths, list) and orient_paths:
            for index, raw in enumerate(orient_paths, start=1):
                add_path(f"Orientação {index}", raw)
        else:
            inline = str(orientations.get("inline") or "").strip()
            rows.append(("Orientações", "texto no TOML" if inline else "—", "OK" if inline else "não informado"))

        project = self.section(cfg, "projeto")
        activity = self.section(cfg, "atividade")
        source_mode = str(activity.get("fonte_dados_atividade") or "").strip()
        ai_mode = bool(activity.get("gerar_dados_atividade_ia", False))
        preset = str(project.get("preset") or "não identificado")
        data_mode_label = "IA" if ai_mode or source_mode == "ia" else (source_mode or "manual")
        rows.append(("Perfil de documento", preset, "OK"))
        rows.append(("Dados acadêmicos", data_mode_label, "OK"))
        # Mantém a denominação anterior para atividades e evita quebrar
        # automações que procuravam esse item no diagnóstico.
        if str(self.section(cfg, "documento").get("tipo_documento") or "").strip().lower() == "atividade":
            rows.append(("Dados da atividade", data_mode_label, "OK"))

        research_source_paths = [
            ("Dados estruturados da pesquisa", "dados_pesquisa_path"),
            ("Arquivo de palavras-chave", "palavras_chave_path"),
            ("Arquivo de hipótese/tese", "hipotese_path"),
        ]
        for label, key in research_source_paths:
            raw = str(research.get(key) or "").strip()
            if raw:
                add_path(label, raw)

        for key in ("global_paths", "institution_paths", "paper_paths", "atividade_paths", "resumo_artigos_paths", "dissertacao_paths", "prisma_paths", "document_paths"):
            values = prompts.get(key, [])
            if isinstance(values, list):
                for index, raw in enumerate(values, start=1):
                    add_path(f"Prompt {key} {index}", raw)
        add_path("Manifesto DOI", local.get("doi_manifest_path"))
        return rows


    def _global_project_context_text(self) -> str:
        """Resumo do TOML global usado pelo fluxo tradicional da pipeline.

        Este contexto NÃO é o mesmo que o artigo final do fluxo PRISMA.
        Ele representa o TOML selecionado para operações gerais: wizard,
        busca PRISMA preliminar, validação e geração tradicional.
        """
        path = self.active_config()
        if not path:
            return (
                "Contexto global da pipeline:\n"
                "Projeto global selecionado: nenhum\n"
                "TOML global: —\n"
                "Saída global: —"
            )
        try:
            cfg = self.load_toml(path)
        except Exception as exc:
            return (
                "Contexto global da pipeline:\n"
                f"TOML global ilegível: {path}\n"
                f"Erro: {exc}"
            )
        project = self.section(cfg, "projeto")
        document = self.section(cfg, "documento")
        title = str(document.get("titulo_trabalho") or self.section(cfg, "atividade").get("titulo_trabalho") or "sem título")
        name = str(project.get("nome") or path.parent.name)
        rows = self.input_rows(path, cfg)
        missing = sum(1 for _label, _item, status in rows if status in {"AUSENTE", "PENDENTE"})
        out_dir = self.cfg_output_dir(path, cfg)
        artifacts = len([p for p in out_dir.glob("*") if p.is_file()]) if out_dir.exists() else 0
        return (
            "Contexto global da pipeline:\n"
            f"Projeto global selecionado: {name}\n"
            f"Título global: {title}\n"
            f"TOML global: {path}\n"
            f"Insumos pendentes do TOML global: {missing}\n"
            f"Artefatos na saída global: {artifacts}\n"
            f"Saída global: {out_dir}"
        )

    def _article_context_text(self) -> str:
        """Resumo do artigo final controlado pelo fluxo PRISMA robusto."""
        raw_dir = str(self.state.get("last_article_dir") or "").strip()
        raw_cfg = str(self.state.get("last_article_cfg") or "").strip()

        if not raw_dir and not raw_cfg:
            return (
                "Contexto do artigo PRISMA:\n"
                "Artigo ativo: nenhum\n"
                "Pasta do artigo: —\n"
                "TOML do artigo final: —\n"
                "Estado estrutural: ainda não criado"
            )

        art_dir = Path(raw_dir).expanduser().resolve() if raw_dir else None
        cfg_art = Path(raw_cfg).expanduser().resolve() if raw_cfg else None
        state_file = art_dir / "artigo_state.json" if art_dir else None
        out_dir = art_dir / "output" if art_dir else None

        pdf_status = "não identificado"
        if out_dir and cfg_art:
            pdf = out_dir / f"{cfg_art.stem}.pdf"
            pdf_status = f"existe — {pdf}" if pdf.exists() else f"não gerado — esperado em {pdf}"
        elif out_dir and out_dir.exists():
            pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            pdf_status = f"existe — {pdfs[0]}" if pdfs else f"não encontrado em {out_dir}"

        state_status = "não criado"
        if state_file:
            state_status = f"existe — {state_file}" if state_file.exists() else f"não criado — esperado em {state_file}"

        return (
            "Contexto do artigo PRISMA:\n"
            f"Artigo ativo: {art_dir if art_dir else 'pasta não definida'}\n"
            f"Pasta do artigo: {art_dir if art_dir else '—'}\n"
            f"TOML do artigo final: {cfg_art if cfg_art else '—'}\n"
            f"Estado estrutural: {state_status}\n"
            f"PDF final: {pdf_status}"
        )

    def status_text(self) -> str:
        return (
            "CENTRAL OPERACIONAL FGV — DOCUMENTOS ACADÊMICOS\n"
            "Fluxo completo: escolher categoria → configurar → conferir → validar → gerar → revisar\n\n"
            "Esta tela separa o TOML global da pipeline do artigo final PRISMA.\n"
            "O projeto global é usado pelos fluxos gerais e pela busca PRISMA preliminar; "
            "o artigo PRISMA tem pasta, TOML final e estado estrutural próprios.\n\n"
            + self._global_project_context_text()
            + "\n\n"
            + self._article_context_text()
        )

    def require_config(self) -> Path | None:
        path = self.active_config()
        if path is None:
            ui.message("Academic Pipeline — projeto necessário", "Não há um TOML ativo. Crie ou selecione um documento antes de continuar.")
        return path

    # ------------------------------------------------------------------
    # Execução e logs
    # ------------------------------------------------------------------

    def _write_log(self, label: str, command: list[str], stdout: str, returncode: int) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{stamp}_{label}.log"
        body = "\n".join(
            [
                "Academic Pipeline TUI — registro operacional",
                f"Data: {datetime.now().isoformat(timespec='seconds')}",
                f"Comando: {' '.join(command)}",
                f"Código de saída: {returncode}",
                "",
                stdout.rstrip(),
                "",
            ]
        )
        log_path.write_text(body, encoding="utf-8")
        return log_path

    def run_command(self, args: list[str], *, label: str, title: str) -> int:
        command = [sys.executable, str(PIPELINE), *args]
        # A execução preserva o terminal limpo; o resultado é devolvido em
        # painel rolável e também fica registrado em output/tui_logs.
        try:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:
            ui.message(title, f"Falha ao iniciar o comando:\n{exc}")
            return 1
        output = proc.stdout or "(o comando não produziu saída textual)"
        log_path = self._write_log(label, command, output, proc.returncode)
        summary = (
            f"Comando concluído com código {proc.returncode}.\n"
            f"Log operacional: {log_path}\n\n"
            f"{'─' * 88}\n{output}"
        )
        ui.message(title, summary, width=126)
        return int(proc.returncode)

    # ------------------------------------------------------------------
    # Projeto e assistente de TOML
    # ------------------------------------------------------------------

    def choose_existing_config(self) -> Path | None:
        configs = self.list_project_configs()
        if not configs:
            ui.message("Academic Pipeline — documentos", "Nenhum TOML foi localizado. Crie um documento novo ou informe o diretório externo do projeto.")
            return None
        values: list[tuple[str, str]] = []
        for path in configs:
            try:
                cfg = self.load_toml(path)
                project = self.section(cfg, "projeto")
                document = self.section(cfg, "documento")
                name = str(project.get("nome") or path.parent.name)
                title = str(document.get("titulo_trabalho") or "sem título")
                preset = str(project.get("preset") or document.get("tipo_documento") or "documento")
                label = f"{name} — {title}\nPerfil: {preset}\n{path}"
            except Exception:
                label = str(path)
            values.append((str(path), label))
        selected = ui.select_one(
            "Academic Pipeline — selecionar documento",
            "Use ↑/↓ para navegar. O TOML escolhido será mantido como documento ativo.",
            values,
            default=str(self.active_config() or configs[0]),
            width=126,
        )
        if not selected:
            return None
        path = Path(str(selected)).resolve()
        self.set_active_config(path)
        ui.message("Academic Pipeline — TOML global da pipeline", f"TOML global da pipeline atualizado:\n{path}")
        return path

    def choose_document_profile(self) -> str | None:
        return ui.select_one(
            "Academic Pipeline — passo 1/8: tipo de documento",
            (
                "Escolha o fluxo que será montado. Depois a TUI solicitará o diretório do projeto, "
                "abrirá as perguntas específicas do perfil e seguirá até a geração final."
            ),
            DOCUMENT_PROFILES,
            default="atividade_local_fgv",
            width=126,
        )

    def choose_project_directory(self) -> Path | None:
        default = str(APP / "projetos" / "novo_documento")
        raw = ui.input_text(
            "Academic Pipeline — passo 2/8: diretório do projeto",
            (
                "Informe a pasta que receberá o TOML, os arquivos auxiliares e, por padrão, a pasta output.\n"
                "Ela pode ser externa ao bundle, por exemplo a pasta da atividade dentro de sua disciplina."
            ),
            default=default,
            path_completion=True,
            only_directories=True,
            width=126,
        )
        if raw is None:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists() and not candidate.is_dir():
            ui.message("Academic Pipeline — diretório", f"O caminho informado não é uma pasta:\n{candidate}")
            return None
        if not candidate.exists() and not ui.confirm(
            "Academic Pipeline — criar diretório",
            f"A pasta ainda não existe. Criar agora?\n{candidate}",
            default=True,
        ):
            return None
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def run_wizard(self, profile_key: str | None = None) -> int:
        profile = profile_key or self.choose_document_profile()
        if not profile:
            return 1
        project_dir = self.choose_project_directory()
        if project_dir is None:
            return 1
        before = {str(path.resolve()) for path in project_dir.glob("*.toml") if path.is_file()}
        command = [
            sys.executable,
            str(TOML_WIZARD),
            "--profile",
            str(profile),
            "--project-dir",
            str(project_dir),
            "--tui-theme",
            "fgv",
        ]
        try:
            rc = subprocess.run(command, cwd=str(ROOT), check=False).returncode
        except Exception as exc:
            ui.message("Academic Pipeline — assistente", f"Não foi possível abrir o assistente:\n{exc}")
            return 1
        candidates = sorted(
            [path.resolve() for path in project_dir.glob("*.toml") if path.is_file() and self.is_project_config(path)],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        new_items = [path for path in candidates if str(path) not in before]
        target = new_items[0] if new_items else (candidates[0] if candidates else None)
        if rc == 0 and target:
            self.set_active_config(target)
            ui.message(
                "Academic Pipeline — passo 6/8: TOML salvo",
                f"Documento configurado. O TOML ativo será usado na validação e na geração.\n\n{target}",
            )
        elif rc == 0:
            ui.message(
                "Academic Pipeline — assistente",
                "O assistente terminou, mas não localizei um TOML válido na pasta escolhida. Verifique o diretório informado.",
            )
        else:
            ui.message("Academic Pipeline — assistente", "O assistente foi encerrado ou cancelado antes de gravar um TOML.")
        return int(rc)

    def menu_project(self) -> None:
        while True:
            choice = ui.menu(
                "Academic Pipeline — documento e configuração",
                self.status_text(),
                [
                    ("new", "[N] Novo documento: escolher tipo e montar TOML", ["n", "1"]),
                    ("select", "[S] Selecionar TOML existente", ["s", "2"]),
                    ("open", "[O] Abrir TOML ativo no aplicativo padrão", ["o", "3"]),
                    ("clear", "[L] Limpar projeto ativo", ["l"]),
                    ("back", "[V] Voltar ao painel", ["v", "0"]),
                ],
            )
            if choice in {None, "back"}:
                return
            if choice == "new":
                self.run_wizard()
            elif choice == "select":
                self.choose_existing_config()
            elif choice == "open":
                path = self.require_config()
                if path:
                    self.open_path(path)
            elif choice == "clear":
                if ui.confirm("Academic Pipeline — TOML global da pipeline", "Remover a referência ao TOML ativo? O arquivo não será excluído.", default=False):
                    self.set_active_config(None)

    # ------------------------------------------------------------------
    # Conferência, validação e produção
    # ------------------------------------------------------------------

    def show_inputs(self) -> bool:
        path = self.require_config()
        if not path:
            return False
        try:
            rows = self.input_rows(path)
        except Exception as exc:
            ui.message("Academic Pipeline — insumos", f"Não foi possível ler o TOML:\n{exc}")
            return False
        lines = [f"TOML ativo: {path}", "", "STATUS   ITEM", "─" * 100]
        missing_corpus = False
        for label, item, status in rows:
            marker = "✓" if status in {"OK", "IA", "interno do perfil"} else ("!" if status in {"AUSENTE", "PENDENTE"} else "·")
            lines.append(f"{marker} {status:<14} {label}: {item}")
            if label in {"Corpus ZIP", "Corpus pasta"} and status in {"AUSENTE", "PENDENTE"}:
                missing_corpus = True
        lines.append("")
        lines.append("Resultado: " + ("corpus pendente; corrija o TOML antes da geração." if missing_corpus else "corpus disponível para validação."))
        ui.message("Academic Pipeline — corpus, orientações e prompts", "\n".join(lines), width=126)
        return not missing_corpus

    def validate_config(self) -> bool:
        path = self.require_config()
        if not path:
            return False
        rc = self.run_command(["--config", str(path), "--check-config"], label="check_config", title="Academic Pipeline — validação preventiva")
        return rc == 0

    def show_prompts(self) -> None:
        path = self.require_config()
        if path:
            self.run_command(["--config", str(path), "--show-prompts"], label="show_prompts", title="Academic Pipeline — diretivas ativas")

    def write_prompt_lock(self) -> None:
        path = self.require_config()
        if path:
            self.run_command(["--config", str(path), "--write-prompt-lock"], label="prompt_lock", title="Academic Pipeline — prompt lock")

    def generate_full(self) -> bool:
        path = self.require_config()
        if not path:
            return False
        try:
            cfg = self.load_toml(path)
        except Exception:
            cfg = {}
        busca = self.section(cfg, "busca_prisma")
        is_external_prisma = bool(busca.get("ativo", False)) and str(busca.get("modo") or "").strip().lower() == "busca_externa"
        prompt = (
            "A geração consultará as bases bibliográficas configuradas, deduplicará os registros e produzirá a planilha de triagem humana. "
            "Ela não decidirá automaticamente a inclusão de estudos nem baixará PDFs. Confirmar agora?"
            if is_external_prisma
            else "A geração completa consultará a IA e produzirá document.json, ORG, PDF/DOCX conforme o TOML, além dos relatórios de controle. Confirmar agora?"
        )
        if not ui.confirm("Academic Pipeline — confirmar geração", prompt, default=False):
            return False
        label = "generate_prisma_search" if is_external_prisma else "generate_activity"
        title = "Academic Pipeline — busca PRISMA externa" if is_external_prisma else "Academic Pipeline — geração completa"
        rc = self.run_command(["--config", str(path)], label=label, title=title)
        return rc == 0

    def import_prisma_triage(self) -> None:
        path = self.require_config()
        if not path:
            return
        try:
            cfg = self.load_toml(path)
        except Exception as exc:
            ui.message("Academic Pipeline — triagem PRISMA", f"Não foi possível ler o TOML:\n{exc}")
            return
        busca = self.section(cfg, "busca_prisma")
        if not (bool(busca.get("ativo", False)) and str(busca.get("modo") or "").strip().lower() == "busca_externa"):
            ui.message(
                "Academic Pipeline — triagem PRISMA",
                "Esta opção exige o perfil relatorio_prisma_busca_orientada_fgv e uma planilha criada pela busca externa.",
            )
            return
        output_dir = self.cfg_output_dir(path, cfg)
        candidates = sorted(output_dir.glob("*.triagem_titulo_resumo.csv"), key=lambda item: item.stat().st_mtime, reverse=True) if output_dir.exists() else []
        default = str(candidates[0]) if candidates else ""
        raw = ui.input_text(
            "Academic Pipeline — importar triagem PRISMA",
            "Selecione a planilha CSV preenchida. Marque INCLUIR ou EXCLUIR nas colunas de decisão e confirme os estudos na coluna incluir_final.",
            default=default,
            path_completion=True,
            allowed_suffixes=(".csv",),
            width=126,
        )
        if not raw:
            return
        self.run_command(
            ["--config", str(path), "--prisma-importar-triagem", str(Path(raw).expanduser())],
            label="import_prisma_triage",
            title="Academic Pipeline — consolidação da triagem PRISMA",
        )

    def rerender_existing(self) -> None:
        path = self.require_config()
        if not path:
            return
        cfg = self.load_toml(path)
        output_dir = self.cfg_output_dir(path, cfg)
        candidates = sorted(output_dir.glob("*.document.json"), key=lambda p: p.stat().st_mtime, reverse=True) if output_dir.exists() else []
        default = str(candidates[0]) if candidates else ""
        raw = ui.input_text(
            "Academic Pipeline — renderizar sem IA",
            "Caminho do document.json existente:",
            default=default,
            path_completion=True,
            width=126,
        )
        if not raw:
            return
        self.run_command(
            ["--config", str(path), "--somente-renderizar", "--document-json", str(Path(raw).expanduser())],
            label="render_existing",
            title="Academic Pipeline — renderização sem IA",
        )

    def recompile_org(self) -> None:
        path = self.require_config()
        if not path:
            return
        cfg = self.load_toml(path)
        output_dir = self.cfg_output_dir(path, cfg)
        candidates = sorted(output_dir.glob("*.org"), key=lambda p: p.stat().st_mtime, reverse=True) if output_dir.exists() else []
        default = str(candidates[0]) if candidates else ""
        raw = ui.input_text(
            "Academic Pipeline — recompilar ORG",
            "Caminho do arquivo ORG:",
            default=default,
            path_completion=True,
            width=126,
        )
        if not raw:
            return
        self.run_command(["--config", str(path), "--recompile", "--org", str(Path(raw).expanduser())], label="recompile_org", title="Academic Pipeline — recompilação")


    # ------------------------------------------------------------------
    # Fluxo PRISMA -> artigo final unificado
    # ------------------------------------------------------------------

    def cfg_research_output_dir(self, config_path: Path, cfg: dict[str, Any] | None = None) -> Path:
        """Diretório do relatório PRISMA, distinto da saída do documento."""
        cfg = cfg if cfg is not None else self.load_toml(config_path)
        paths = self.section(cfg, "paths")
        project = self.section(cfg, "projeto")
        raw_base = paths.get("research_output_dir") or "output_pesquisa"
        base = self.resolve_config_path(config_path, raw_base) or (config_path.parent / "output_pesquisa")
        prefix = str(paths.get("research_prefix") or f"relatorio_prisma_{project.get('nome') or config_path.stem}").strip()
        return base / prefix if bool(paths.get("create_research_subdir", True)) else base

    def _run_python_file(self, script: Path, args: list[str], *, label: str, title: str) -> int:
        if not script.exists():
            ui.message(title, f"Script não encontrado:\n{script}")
            return 1
        command = [sys.executable, str(script), *args]
        try:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:
            ui.message(title, f"Falha ao iniciar o comando:\n{exc}")
            return 1
        output = proc.stdout or "(o comando não produziu saída textual)"
        log_path = self._write_log(label, command, output, proc.returncode)
        summary = f"Comando concluído com código {proc.returncode}.\nLog operacional: {log_path}\n\n{'─' * 88}\n{output}"
        ui.message(title, summary, width=126)
        return int(proc.returncode)

    def _input_int(self, title: str, text: str, default: int) -> int | None:
        raw = ui.input_text(title, text, default=str(default), width=126)
        if raw is None:
            return None
        raw = str(raw).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            ui.message(title, f"Valor inteiro inválido: {raw}")
            return None
        if value <= 0:
            ui.message(title, "O valor deve ser positivo.")
            return None
        return value


    def _article_workflow(self, *, require_art_dir: bool = False):
        if ArticleWorkflow is None:
            return None
        raw = str(self.state.get("last_article_dir") or "").strip()
        if not raw:
            if require_art_dir:
                ui.message("Artigo PRISMA — estado", "Defina primeiro a pasta do artigo no briefing ou na geração final.")
            return None
        art_dir = Path(raw).expanduser().resolve()
        cfg_raw = str(self.state.get("last_article_cfg") or "").strip()
        cfg_art = Path(cfg_raw).expanduser().resolve() if cfg_raw else None
        prisma_cfg = self.active_config()
        try:
            return ArticleWorkflow(art_dir, cfg_art=cfg_art, prisma_cfg=prisma_cfg)
        except Exception as exc:
            ui.message("Artigo PRISMA — estado", f"Não foi possível carregar o estado do fluxo:\n{exc}")
            return None

    def _article_workflow_status_text(self) -> str:
        wf = self._article_workflow(require_art_dir=False)
        if wf is None:
            return "Status estrutural: pasta de artigo ainda não definida. Comece pelo briefing."
        try:
            return wf.format_status()
        except Exception as exc:
            return f"Status estrutural indisponível: {exc}"

    def _article_workflow_mark_ok(self, stage: str, *, evidence: list[str] | None = None, message: str = "") -> None:
        wf = self._article_workflow(require_art_dir=False)
        if wf is None:
            return
        try:
            wf.mark_stage_ok(stage, evidence=evidence or [], message=message)
        except Exception:
            return

    def _article_workflow_can_run_or_message(self, stage: str, title: str) -> bool:
        wf = self._article_workflow(require_art_dir=True)
        if wf is None:
            return True
        try:
            ok, msg = wf.can_run(stage)
        except Exception as exc:
            ui.message(title, f"Não foi possível validar bloqueios do fluxo:\n{exc}")
            return False
        if not ok:
            ui.message(title, msg + "\n\nUse o menu do artigo para concluir as etapas anteriores ou atualize o diagnóstico.")
            return False
        return True

    def mark_prisma_xlsx_reviewed(self) -> None:
        wf = self._article_workflow(require_art_dir=True)
        if wf is None:
            return
        try:
            wf.refresh_from_files()
            candidates = wf.find_prisma_artifacts(["*.curadoria_ia_referencias.xlsx"])
        except Exception as exc:
            ui.message("Artigo PRISMA — revisão humana", f"Não foi possível localizar XLSX de curadoria:\n{exc}")
            return
        if not candidates:
            ui.message("Artigo PRISMA — revisão humana", "Nenhum XLSX de curadoria IA foi localizado.")
            return
        selected = ui.select_one(
            "Artigo PRISMA — confirmar revisão humana",
            "Selecione o XLSX que você revisou manualmente e salvou. A partir daqui, o PRISMA final fica desbloqueado.",
            [(str(p), f"{p.name}\n{p}") for p in candidates],
            default=str(candidates[0]),
            width=126,
        )
        if not selected:
            return
        xlsx = Path(str(selected)).expanduser().resolve()
        if not ui.confirm("Artigo PRISMA — revisão humana", f"Confirmar que este XLSX já foi revisado manualmente e salvo?\n{xlsx}", default=False):
            return
        try:
            wf.mark_human_review(xlsx)
        except Exception as exc:
            ui.message("Artigo PRISMA — revisão humana", f"Falha ao registrar revisão humana:\n{exc}")
            return
        ui.message("Artigo PRISMA — revisão humana", "Revisão humana registrada. O PRISMA final está desbloqueado.")

    def create_or_edit_briefing(self) -> None:
        default_dir = str(Path(self.state.get("last_article_dir") or Path.home()).expanduser())
        raw_dir = ui.input_text(
            "Artigo PRISMA — briefing",
            "Informe a pasta do artigo ou da atividade. O briefing será criado/aberto nela.",
            default=default_dir,
            path_completion=True,
            only_directories=True,
            width=126,
        )
        if not raw_dir:
            return
        art_dir = Path(raw_dir).expanduser().resolve()
        art_dir.mkdir(parents=True, exist_ok=True)
        self.state["last_article_dir"] = str(art_dir)
        self._save_state()
        briefing = art_dir / "briefing_artigo.txt"
        if not briefing.exists():
            briefing.write_text(
                "Tema:\n\nObjetivo:\n\nProblema de pesquisa:\n\nPergunta norteadora:\n\nRecorte temporal:\n\nRecorte institucional/geográfico:\n\nPalavras-chave:\n\nBases desejadas:\n\nCritérios de inclusão:\n\nCritérios de exclusão:\n\nProduto final esperado:\n\nPadrão de formatação:\n",
                encoding="utf-8",
            )
        self._article_workflow_mark_ok("briefing", evidence=[str(briefing)], message="Briefing criado/aberto para preenchimento.")
        self.open_path(briefing)

    def run_prisma_preliminary_search(self) -> None:
        path = self.require_config()
        if not path:
            return
        if not self.validate_config():
            ui.message("Artigo PRISMA — validação", "Corrija o TOML antes de rodar a busca PRISMA preliminar.")
            return
        if self.generate_full():
            self._article_workflow_mark_ok("prisma_preliminar", message="Busca PRISMA preliminar executada pela TUI.")

    def _curadoria_prompt_default(self, config_path: Path, research_out: Path) -> str:
        candidates: list[Path] = []
        candidates.extend(sorted(config_path.parent.glob("prompt_curadoria*.y*ml")))
        candidates.extend(sorted(research_out.glob("prompt_curadoria*.y*ml")) if research_out.exists() else [])
        raw_last = str(self.state.get("last_curadoria_prompt") or "").strip()
        if raw_last:
            candidates.insert(0, Path(raw_last).expanduser())
        for p in candidates:
            if p.exists() and p.is_file():
                return str(p.resolve())
        return ""

    def run_prisma_ai_cut(self) -> None:
        path = self.require_config()
        if not path:
            return
        try:
            cfg = self.load_toml(path)
            research_out = self.cfg_research_output_dir(path, cfg)
        except Exception as exc:
            ui.message("Artigo PRISMA — curadoria IA", f"Não foi possível calcular a saída PRISMA:\n{exc}")
            return
        prompt_default = self._curadoria_prompt_default(path, research_out)
        prompt = ui.input_text(
            "Artigo PRISMA — XLSX cut pela IA",
            "Prompt YAML de curadoria. Deixe em branco para usar o default interno do script.",
            default=prompt_default,
            path_completion=True,
            allowed_suffixes=(".yaml", ".yml"),
            width=126,
        )
        if prompt is None:
            return
        max_incluir = self._input_int("Artigo PRISMA — curadoria", "Número máximo de referências a incluir no XLSX cut:", int(self.state.get("last_target_n") or 20))
        if max_incluir is None:
            return
        top_n = self._input_int("Artigo PRISMA — curadoria", "Número de candidatos para a curadoria IA:", int(self.state.get("last_top_n_candidatos") or 90))
        if top_n is None:
            return
        limiar = self._input_int("Artigo PRISMA — curadoria", "Limiar mínimo de inclusão sugerida:", int(self.state.get("last_limiar_minimo") or 45))
        if limiar is None:
            return
        self.state["last_target_n"] = int(max_incluir)
        self.state["last_top_n_candidatos"] = int(top_n)
        self.state["last_limiar_minimo"] = int(limiar)
        if str(prompt).strip():
            self.state["last_curadoria_prompt"] = str(Path(prompt).expanduser().resolve())
        self._save_state()
        args = [
            "--config", str(path),
            "--prisma-curadoria-ia",
            "--prisma-curadoria-out-dir", str(research_out),
            "--prisma-curadoria-max-incluir", str(max_incluir),
            "--prisma-curadoria-top-n-candidatos", str(top_n),
            "--prisma-curadoria-limiar-minimo", str(limiar),
        ]
        if str(prompt).strip():
            args.extend(["--prisma-curadoria-prompt", str(Path(prompt).expanduser().resolve())])
        rc = self.run_command(args, label="prisma_curadoria_ia", title="Artigo PRISMA — XLSX cut pela IA")
        if rc == 0:
            self._article_workflow_mark_ok("xlsx_cut", message="XLSX cut gerado pela curadoria IA.")

    def open_prisma_review_xlsx(self) -> None:
        path = self.require_config()
        if not path:
            return
        cfg = self.load_toml(path)
        research_out = self.cfg_research_output_dir(path, cfg)
        candidates: list[Path] = []
        if research_out.exists():
            for pattern in ["*.curadoria_ia_referencias.xlsx", "*.triagem_titulo_resumo.xlsx", "*.matriz_estudos_incluidos.xlsx"]:
                candidates.extend(research_out.glob(pattern))
        candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            ui.message("Artigo PRISMA — revisão XLSX", f"Nenhum XLSX de curadoria/triagem encontrado em:\n{research_out}")
            return
        selected = ui.select_one(
            "Artigo PRISMA — revisão humana do XLSX",
            "Abra a planilha, revise as decisões de inclusão/exclusão e salve antes de continuar.",
            [(str(p), f"{p.name}\n{p}") for p in candidates],
            default=str(candidates[0]),
            width=126,
        )
        if selected:
            self.open_path(Path(str(selected)))

    def generate_prisma_final_from_review(self) -> None:
        path = self.require_config()
        if not path:
            return
        if not self._article_workflow_can_run_or_message("prisma_final", "Artigo PRISMA — PRISMA final"):
            return
        cfg = self.load_toml(path)
        research_out = self.cfg_research_output_dir(path, cfg)
        candidates = sorted(research_out.glob("*.curadoria_ia_referencias.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True) if research_out.exists() else []
        default = str(candidates[0]) if candidates else ""
        raw = ui.input_text(
            "Artigo PRISMA — gerar PRISMA final",
            "Selecione o XLSX revisado. O sistema reexportará para triagem_humana.csv e importará no PRISMA final.",
            default=default,
            path_completion=True,
            allowed_suffixes=(".xlsx", ".csv"),
            width=126,
        )
        if not raw:
            return
        reviewed = Path(raw).expanduser().resolve()
        rc = self.run_command(
            ["--config", str(path), "--prisma-curadoria-reexportar-xlsx", "--prisma-curadoria-input", str(reviewed), "--prisma-curadoria-out-dir", str(research_out)],
            label="prisma_reexportar_xlsx",
            title="Artigo PRISMA — reexportar XLSX revisado",
        )
        if rc != 0:
            return
        rc2 = self.run_command(
            ["--config", str(path), "--prisma-curadoria-importar", "--prisma-curadoria-out-dir", str(research_out)],
            label="prisma_final_importar",
            title="Artigo PRISMA — PRISMA final",
        )
        if rc2 == 0:
            self._article_workflow_mark_ok("prisma_final", message="PRISMA final importado/consolidado a partir do XLSX revisado.")

    def _ask_article_final_settings(self) -> tuple[Path, Path, int, int, int] | None:
        default_dir = str(self.state.get("last_article_dir") or Path.home())
        raw_dir = ui.input_text(
            "Artigo final — pasta do artigo",
            "Informe a pasta do artigo final. Ela deve conter o TOML do artigo e receberá output/. Também será usada para full text/corpus.",
            default=default_dir,
            path_completion=True,
            only_directories=True,
            width=126,
        )
        if not raw_dir:
            return None
        art_dir = Path(raw_dir).expanduser().resolve()
        if not art_dir.exists():
            if not ui.confirm("Artigo final — criar pasta", f"A pasta não existe. Criar?\n{art_dir}", default=True):
                return None
            art_dir.mkdir(parents=True, exist_ok=True)
        tomls = sorted(art_dir.glob("*.toml"), key=lambda p: p.stat().st_mtime, reverse=True)
        default_cfg = str(self.state.get("last_article_cfg") or (tomls[0] if tomls else art_dir / "artigo_final.toml"))
        raw_cfg = ui.input_text(
            "Artigo final — TOML do artigo",
            "Informe o TOML do artigo final. Ele pode ter sido gerado/congelado a partir do PRISMA final.",
            default=default_cfg,
            path_completion=True,
            allowed_suffixes=(".toml",),
            width=126,
        )
        if not raw_cfg:
            return None
        cfg_art = Path(raw_cfg).expanduser().resolve()
        target_n = self._input_int("Artigo final — parâmetros", "Quantidade alvo de estudos full text:", int(self.state.get("last_target_n") or 20))
        if target_n is None:
            return None
        min_palavras = self._input_int("Artigo final — parâmetros", "Mínimo de palavras do artigo:", int(self.state.get("last_min_palavras") or 8500))
        if min_palavras is None:
            return None
        chars_pdf = self._input_int("Artigo final — parâmetros", "Caracteres por PDF full text na preparação do corpus:", int(self.state.get("last_chars_por_pdf") or 18000))
        if chars_pdf is None:
            return None
        self.state["last_article_dir"] = str(art_dir)
        self.state["last_article_cfg"] = str(cfg_art)
        self.state["last_target_n"] = int(target_n)
        self.state["last_min_palavras"] = int(min_palavras)
        self.state["last_chars_por_pdf"] = int(chars_pdf)
        self._save_state()
        wf = self._article_workflow(require_art_dir=False)
        if wf is not None:
            try:
                wf.refresh_from_files()
            except Exception:
                pass
        return art_dir, cfg_art, target_n, min_palavras, chars_pdf

    def generate_article_final_unified(self) -> None:
        settings = self._ask_article_final_settings()
        if settings is None:
            return
        art_dir, cfg_art, target_n, min_palavras, chars_pdf = settings
        if not cfg_art.exists():
            ui.message("Artigo final — TOML ausente", f"O TOML informado não existe:\n{cfg_art}\n\nGere o TOML do artigo a partir do PRISMA final antes de continuar.")
            return
        if not self._article_workflow_can_run_or_message("fulltext", "Artigo final — gerador unificado"):
            return
        if not ui.confirm("Artigo final — gerar PDF", "Executar o gerador unificado agora? Esta etapa pode demorar.", default=True):
            return
        rc = self._run_python_file(
            ARTICLE_FINALIZER,
            ["--art-dir", str(art_dir), "--cfg-art", str(cfg_art), "--target-n", str(target_n), "--min-palavras", str(min_palavras), "--chars-por-pdf", str(chars_pdf), "--quiet"],
            label="artigo_final_unificado",
            title="Artigo final — gerador unificado",
        )
        if rc == 0:
            wf = self._article_workflow(require_art_dir=False)
            if wf is not None:
                try:
                    wf.refresh_from_files()
                except Exception:
                    pass
            pdf = art_dir / "output" / f"{cfg_art.stem}.pdf"
            msg = f"PDF final gerado com sucesso.\n\nEsperado em:\n{pdf}"
            ui.message("Artigo final — concluído", msg, width=126)
            if pdf.exists() and ui.confirm("Artigo final — abrir PDF", "Abrir o PDF final agora?", default=True):
                self.open_path(pdf)

    def open_article_output_folder(self) -> None:
        raw = str(self.state.get("last_article_dir") or "").strip()
        if not raw:
            settings = self._ask_article_final_settings()
            if settings is None:
                return
            art_dir = settings[0]
        else:
            art_dir = Path(raw).expanduser().resolve()
        out = art_dir / "output"
        if not out.exists():
            ui.message("Artigo final — saída", f"Pasta output ainda não existe:\n{out}")
            return
        self.open_path(out)


    def menu_prisma_article_flow(self) -> None:
        while True:
            path = self.active_config()
            prisma_context = (
                f"TOML PRISMA preliminar/global selecionado: {path}"
                if path
                else "TOML PRISMA preliminar/global selecionado: nenhum"
            )
            description = (
                "Fluxo metodológico para artigo PRISMA até PDF final.\n"
                "O sistema mantém estado persistente, valida contratos de saída e bloqueia etapas críticas fora de ordem.\n\n"
                "Contexto de pesquisa PRISMA preliminar/global:\n"
                f"{prisma_context}\n\n"
                + self._article_context_text()
                + "\n\n"
                + self._article_workflow_status_text()
            )
            choice = ui.menu(
                "Academic Pipeline — Artigo PRISMA até PDF final",
                description,
                [
                    ("briefing", "[1] Criar/abrir briefing do artigo", ["1", "b"]),
                    ("toml", "[2] Gerar TOML PRISMA preliminar pelo assistente", ["2"]),
                    ("select", "[3] Selecionar TOML PRISMA preliminar/global existente", ["3", "s"]),
                    ("prelim", "[4] Rodar pesquisa PRISMA preliminar", ["4", "p"]),
                    ("cut", "[5] Gerar XLSX cut de referências pela IA", ["5", "x"]),
                    ("openxlsx", "[6] Abrir XLSX para revisão humana", ["6", "r"]),
                    ("markreview", "[7] Confirmar que o XLSX foi revisado manualmente", ["7", "m"]),
                    ("finalprisma", "[8] Gerar PRISMA final a partir do XLSX revisado", ["8", "f"]),
                    ("article", "[9] Rodar gerador unificado e abrir PDF final", ["9", "g"]),
                    ("out", "[A] Abrir pasta output do artigo final", ["a", "o"]),
                    ("back", "[0] Voltar ao painel", ["0", "v", "q"]),
                ],
                width=126,
            )
            if choice in {None, "back"}:
                return
            if choice == "briefing":
                self.create_or_edit_briefing()
            elif choice == "toml":
                self.run_wizard("relatorio_prisma_busca_orientada_fgv")
            elif choice == "select":
                self.choose_existing_config()
            elif choice == "prelim":
                self.run_prisma_preliminary_search()
            elif choice == "cut":
                self.run_prisma_ai_cut()
            elif choice == "openxlsx":
                self.open_prisma_review_xlsx()
            elif choice == "markreview":
                self.mark_prisma_xlsx_reviewed()
            elif choice == "finalprisma":
                self.generate_prisma_final_from_review()
            elif choice == "article":
                self.generate_article_final_unified()
            elif choice == "out":
                self.open_article_output_folder()

    # ------------------------------------------------------------------
    # Saídas e integração com desktop
    # ------------------------------------------------------------------

    def open_path(self, path: Path) -> None:
        if not path.exists():
            ui.message("Academic Pipeline — abrir", f"Caminho inexistente:\n{path}")
            return
        opener = shutil.which("xdg-open")
        if not opener:
            ui.message("Academic Pipeline — abrir", f"Não encontrei xdg-open. Abra manualmente:\n{path}")
            return
        try:
            subprocess.Popen([opener, str(path)], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            ui.message("Academic Pipeline — abrir", f"Não foi possível abrir:\n{exc}")
            return
        ui.message("Academic Pipeline — abrir", f"Solicitado ao sistema abrir:\n{path}")

    def list_output_files(self) -> tuple[Path | None, list[Path]]:
        path = self.require_config()
        if not path:
            return None, []
        try:
            output_dir = self.cfg_output_dir(path)
        except Exception as exc:
            ui.message("Academic Pipeline — saídas", f"Não foi possível calcular o diretório de saída:\n{exc}")
            return None, []
        files = sorted([p for p in output_dir.glob("*") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True) if output_dir.exists() else []
        return output_dir, files

    def menu_outputs(self) -> None:
        while True:
            output_dir, files = self.list_output_files()
            if output_dir is None:
                return
            description = [f"Diretório de saída: {output_dir}", ""]
            if files:
                description.append("Artefatos recentes:")
                for file in files[:12]:
                    stamp = datetime.fromtimestamp(file.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
                    description.append(f"• {file.name} — {file.stat().st_size / 1024:.1f} KiB — {stamp}")
            else:
                description.append("Nenhum artefato gerado ainda.")
            choice = ui.menu(
                "Academic Pipeline — saídas, relatórios e logs",
                "\n".join(description),
                [
                    ("folder", "[P] Abrir pasta de saída", ["p", "1"]),
                    ("artifact", "[A] Selecionar e abrir artefato", ["a", "2"]),
                    ("quality", "[Q] Abrir relatório de qualidade", ["q", "3"]),
                    ("compliance", "[C] Abrir relatório de conformidade", ["c", "4"]),
                    ("back", "[V] Voltar ao painel", ["v", "0"]),
                ],
            )
            if choice in {None, "back"}:
                return
            if choice == "folder":
                self.open_path(output_dir)
            elif choice == "artifact":
                if not files:
                    ui.message("Academic Pipeline — artefatos", "Nenhum artefato disponível.")
                    continue
                values = [(str(file), file.name) for file in files]
                selected = ui.select_one("Academic Pipeline — abrir artefato", "Selecione o arquivo a abrir.", values, width=126)
                if selected:
                    self.open_path(Path(str(selected)))
            elif choice in {"quality", "compliance"}:
                token = "quality_report" if choice == "quality" else "compliance"
                matches = [p for p in files if token in p.name.lower() and p.suffix.lower() in {".md", ".txt", ".json"}]
                if not matches:
                    ui.message("Academic Pipeline — relatório", "O relatório ainda não foi encontrado na pasta de saída.")
                else:
                    self.open_path(matches[0])

    # ------------------------------------------------------------------
    # Fluxo guiado e painel
    # ------------------------------------------------------------------

    def run_configured_flow(self) -> None:
        """Etapas 7 e 8: conferir, validar, gerar e abrir os resultados."""
        path = self.active_config()
        if not path:
            return
        inputs_ok = self.show_inputs()
        if not inputs_ok:
            ui.message(
                "Academic Pipeline — passo 7/8: insumos pendentes",
                "O TOML possui caminhos pendentes ou inexistentes. Reabra o assistente para corrigir o projeto antes de validar.",
            )
            return
        if not ui.confirm(
            "Academic Pipeline — passo 7/8: validar",
            "Insumos conferidos. Executar a validação preventiva do TOML agora?",
            default=True,
        ):
            return
        if not self.validate_config():
            ui.message(
                "Academic Pipeline — validação",
                "A validação encontrou pendências. Corrija o TOML antes de gerar o documento.",
            )
            return
        if not ui.confirm(
            "Academic Pipeline — passo 8/8: gerar",
            "Validação concluída. Iniciar a geração final com IA e produzir os arquivos configurados?",
            default=True,
        ):
            return
        if not self.generate_full():
            return
        output_dir, files = self.list_output_files()
        names = "\n".join(f"✓ {file.name}" for file in files) if files else "Nenhum artefato foi localizado no diretório esperado. Consulte o log operacional."
        ui.message(
            "Academic Pipeline — fluxo concluído",
            f"Documento gerado.\n\nDiretório de saída:\n{output_dir}\n\nArtefatos:\n{names}",
            width=126,
        )
        if output_dir and ui.confirm("Academic Pipeline — revisar", "Abrir a pasta de saída agora?", default=True):
            self.open_path(output_dir)

    def run_new_document_flow(self) -> None:
        """Fluxo completo para um novo documento, sem reutilizar TOML ativo."""
        rc = self.run_wizard()
        if rc == 0 and self.active_config() is not None:
            self.run_configured_flow()

    def run_guided_flow(self) -> None:
        """Retoma o fluxo de um TOML existente ou permite selecioná-lo."""
        if self.active_config() is None:
            action = ui.select_one(
                "Academic Pipeline — documento existente",
                "Escolha como retomar um fluxo já configurado.",
                [
                    ("select", "[S] Selecionar TOML existente"),
                    ("new", "[N] Criar novo documento"),
                    ("cancel", "[C] Cancelar"),
                ],
                default="select",
            )
            if action == "new":
                self.run_new_document_flow()
                return
            if action == "select":
                self.choose_existing_config()
            else:
                return
        self.run_configured_flow()


    def main_menu(self) -> int:
        """Menu principal reorganizado por categorias de uso.

        A intenção é separar claramente:
        - artigo PRISMA completo, que é um fluxo metodológico próprio;
        - documentos acadêmicos gerais, que usam o assistente tradicional;
        - administração de TOML/insumos;
        - validação, produção técnica e saídas.
        """
        while True:
            description = (
                self.status_text()
                + "\n\n"
                + "Escolha a categoria do trabalho.\n"
                + "Para artigo com revisão estruturada/PRISMA, use a primeira opção. "
                + "Para atividades, resumos, papers locais e dissertações, use documentos acadêmicos gerais."
            )
            choice = ui.menu(
                "Academic Pipeline — Central Operacional FGV",
                description,
                [
                    ("prisma_article", "[1] Artigos PRISMA e revisão estruturada — briefing → PRISMA → PDF final", ["1", "a"]),
                    ("general_documents", "[2] Documentos acadêmicos gerais — atividade, resumo, paper, dissertação", ["2", "d"]),
                    ("configuration", "[3] Projetos, TOML e insumos — criar, selecionar, abrir e conferir", ["3", "p"]),
                    ("validation", "[4] Validação e prompts — checar TOML, diretivas e prompt lock", ["4", "v"]),
                    ("production", "[5] Produção técnica — gerar, importar triagem, renderizar ou recompilar", ["5", "g"]),
                    ("outputs", "[6] Saídas e relatórios — abrir PDF, DOCX, ORG, logs e pastas", ["6", "o"]),
                    ("exit", "[0] Sair", ["0", "q"]),
                ],
                width=126,
            )
            if choice in {None, "exit"}:
                return 0

            if choice == "prisma_article":
                self.menu_prisma_article_flow()

            elif choice == "general_documents":
                while True:
                    action = ui.menu(
                        "Academic Pipeline — documentos acadêmicos gerais",
                        (
                            self.status_text()
                            + "\n\n"
                            + "Use este grupo para produtos que não exigem o ciclo completo PRISMA→XLSX→full text.\n"
                            + "Exemplos: atividade de disciplina, resumo de artigos, paper local, dissertação local ou renderização de document.json."
                        ),
                        [
                            ("new", "[1] Criar novo documento pelo assistente", ["1", "n"]),
                            ("resume", "[2] Retomar TOML existente: conferir, validar e gerar", ["2", "r"]),
                            ("profiles", "[3] Ver perfis disponíveis no assistente", ["3", "f"]),
                            ("back", "[0] Voltar ao menu principal", ["0", "q", "v"]),
                        ],
                        width=126,
                    )
                    if action in {None, "back"}:
                        break
                    if action == "new":
                        self.run_new_document_flow()
                    elif action == "resume":
                        self.run_guided_flow()
                    elif action == "profiles":
                        lines = ["Perfis disponíveis no assistente:", ""]
                        for key, label in DOCUMENT_PROFILES:
                            clean = re.sub(r"^\[[0-9]+\]\s*", "", label)
                            lines.append(f"• {key}: {clean}")
                        ui.message("Academic Pipeline — perfis de documento", "\n".join(lines), width=126)

            elif choice == "configuration":
                while True:
                    action = ui.menu(
                        "Academic Pipeline — projetos, TOML e insumos",
                        self.status_text(),
                        [
                            ("project", "[1] Criar/selecionar/abrir TOML", ["1", "t"]),
                            ("inputs", "[2] Conferir corpus, orientações, prompts e caminhos", ["2", "i"]),
                            ("select", "[3] Selecionar TOML existente diretamente", ["3", "s"]),
                            ("back", "[0] Voltar ao menu principal", ["0", "q", "v"]),
                        ],
                        width=126,
                    )
                    if action in {None, "back"}:
                        break
                    if action == "project":
                        self.menu_project()
                    elif action == "inputs":
                        self.show_inputs()
                    elif action == "select":
                        self.choose_existing_config()

            elif choice == "validation":
                while True:
                    action = ui.menu(
                        "Academic Pipeline — validação e prompts",
                        self.status_text(),
                        [
                            ("check", "[1] Validar configuração/TOML", ["1", "v"]),
                            ("prompts", "[2] Ver prompts e diretivas ativos", ["2", "p"]),
                            ("lock", "[3] Gerar prompt lock", ["3", "l"]),
                            ("back", "[0] Voltar ao menu principal", ["0", "q"]),
                        ],
                        width=126,
                    )
                    if action in {None, "back"}:
                        break
                    if action == "check":
                        self.validate_config()
                    elif action == "prompts":
                        self.show_prompts()
                    elif action == "lock":
                        self.write_prompt_lock()

            elif choice == "production":
                while True:
                    action = ui.menu(
                        "Academic Pipeline — produção técnica",
                        (
                            "Operações de execução direta.\n"
                            "Em fluxos PRISMA completos, prefira a categoria 1, que apresenta as etapas na ordem metodológica."
                        ),
                        [
                            ("full", "[1] Gerar documento completo ou executar busca PRISMA do TOML ativo", ["1", "g"]),
                            ("triage", "[2] Importar planilha de triagem PRISMA preenchida", ["2", "t"]),
                            ("render", "[3] Somente renderizar document.json existente", ["3", "r"]),
                            ("recompile", "[4] Recompilar arquivo ORG", ["4", "o"]),
                            ("back", "[0] Voltar ao menu principal", ["0", "q"]),
                        ],
                        width=126,
                    )
                    if action in {None, "back"}:
                        break
                    if action == "full":
                        self.generate_full()
                    elif action == "triage":
                        self.import_prisma_triage()
                    elif action == "render":
                        self.rerender_existing()
                    elif action == "recompile":
                        self.recompile_org()

            elif choice == "outputs":
                self.menu_outputs()

# Compatibilidade com os testes e integrações da rc10.7.39. A implementação
# agora é visual, mas preserva a classe pública que era usada por automações.
TerminalUI = AcademicPipelineTUI


def run_tui(*, no_clear: bool = False) -> int:
    try:
        return AcademicPipelineTUI(no_clear=no_clear).main_menu()
    except ui.TUIUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Central operacional visual do Academic Pipeline")
    parser.add_argument("--no-clear", action="store_true", help="Compatibilidade; a TUI visual já controla a tela")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_tui(no_clear=bool(args.no_clear))


if __name__ == "__main__":
    raise SystemExit(main())
