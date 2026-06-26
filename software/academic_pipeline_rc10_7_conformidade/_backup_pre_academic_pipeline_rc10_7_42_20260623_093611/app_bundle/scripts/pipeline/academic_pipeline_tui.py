#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central operacional visual do Academic Pipeline.

Esta TUI segue a convenção da Administração UAC: diálogos ``prompt_toolkit``
em tela inteira, lista vertical navegável, atalhos de teclado, confirmação
explícita antes de ações dispendiosas e campos de caminho com ``Tab`` para
conclusão automática. Não é uma GUI de desktop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import academic_pipeline_tui_widgets as ui


HERE = Path(__file__).resolve()
APP = HERE.parents[2]
ROOT = APP.parent
PIPELINE = HERE.with_name("academic_pipeline_rc10.py")
TOML_WIZARD = HERE.with_name("academic_pipeline_toml_generator_interativo.py")
STATE_PATH = APP / ".academic_pipeline_tui_state.json"
LOG_DIR = APP / "output" / "tui_logs"


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
            self.state["active_config"] = str(path.resolve())
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
    def is_activity_config(cls, path: Path) -> bool:
        try:
            cfg = cls.load_toml(path)
        except Exception:
            return False
        project = cls.section(cfg, "projeto")
        document = cls.section(cfg, "documento")
        preset = str(project.get("preset") or "").strip().lower()
        doc_type = str(document.get("tipo_documento") or "").strip().lower()
        return preset == "atividade_local_fgv" or doc_type == "atividade"

    def list_activity_configs(self) -> list[Path]:
        base = APP / "projetos"
        if not base.exists():
            return []
        values = [p.resolve() for p in base.rglob("*.toml") if p.is_file() and self.is_activity_config(p)]
        return sorted(values, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)

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

        input_zip = str(local.get("input_zip") or "").strip()
        input_dir = str(local.get("input_dir") or "").strip()
        add_path("Corpus ZIP", input_zip, required=not input_dir)
        add_path("Corpus pasta", input_dir, required=not input_zip)

        orient_paths = orientations.get("paths", [])
        if isinstance(orient_paths, list) and orient_paths:
            for index, raw in enumerate(orient_paths, start=1):
                add_path(f"Orientação {index}", raw)
        else:
            inline = str(orientations.get("inline") or "").strip()
            rows.append(("Orientações", "texto no TOML" if inline else "—", "OK" if inline else "não informado"))

        activity = self.section(cfg, "atividade")
        source_mode = str(activity.get("fonte_dados_atividade") or "").strip()
        ai_mode = bool(activity.get("gerar_dados_atividade_ia", False))
        rows.append(("Dados da atividade", "IA" if ai_mode or source_mode == "ia" else (source_mode or "manual"), "OK"))

        for key in ("global_paths", "institution_paths", "atividade_paths", "document_paths"):
            values = prompts.get(key, [])
            if isinstance(values, list):
                for index, raw in enumerate(values, start=1):
                    add_path(f"Prompt {key} {index}", raw)
        add_path("Manifesto DOI", local.get("doi_manifest_path"))
        return rows

    def status_text(self) -> str:
        path = self.active_config()
        if not path:
            return (
                "CENTRAL OPERACIONAL DE ATIVIDADES\n"
                "Fluxo: configurar → conferir → validar → gerar → revisar\n\n"
                "Projeto ativo: nenhum. Comece por “Criar/selecionar atividade”."
            )
        try:
            cfg = self.load_toml(path)
        except Exception as exc:
            return f"CENTRAL OPERACIONAL DE ATIVIDADES\n\nTOML ativo ilegível: {path}\nErro: {exc}"
        project = self.section(cfg, "projeto")
        document = self.section(cfg, "documento")
        title = str(document.get("titulo_trabalho") or self.section(cfg, "atividade").get("titulo_trabalho") or "sem título")
        name = str(project.get("nome") or path.parent.name)
        rows = self.input_rows(path, cfg)
        missing = sum(1 for _label, _item, status in rows if status in {"AUSENTE", "PENDENTE"})
        out_dir = self.cfg_output_dir(path, cfg)
        artifacts = len([p for p in out_dir.glob("*") if p.is_file()]) if out_dir.exists() else 0
        return (
            "CENTRAL OPERACIONAL DE ATIVIDADES\n"
            "Fluxo: configurar → conferir → validar → gerar → revisar\n\n"
            f"Projeto ativo: {name}\n"
            f"Título: {title}\n"
            f"TOML: {path}\n"
            f"Insumos pendentes: {missing}\n"
            f"Artefatos na saída: {artifacts}\n"
            f"Saída: {out_dir}"
        )

    def require_config(self) -> Path | None:
        path = self.active_config()
        if path is None:
            ui.message("Academic Pipeline — projeto necessário", "Não há um TOML ativo. Crie ou selecione uma atividade antes de continuar.")
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
        configs = self.list_activity_configs()
        if not configs:
            ui.message("Academic Pipeline — atividades", "Nenhum TOML de atividade foi localizado em app_bundle/projetos.")
            return None
        values: list[tuple[str, str]] = []
        for path in configs:
            try:
                cfg = self.load_toml(path)
                project = self.section(cfg, "projeto")
                document = self.section(cfg, "documento")
                name = str(project.get("nome") or path.parent.name)
                title = str(document.get("titulo_trabalho") or "sem título")
                label = f"{name} — {title}\n{path}"
            except Exception:
                label = str(path)
            values.append((str(path), label))
        selected = ui.select_one(
            "Academic Pipeline — selecionar atividade",
            "Use ↑/↓ para navegar. O TOML escolhido será mantido como projeto ativo.",
            values,
            default=str(self.active_config() or configs[0]),
            width=126,
        )
        if not selected:
            return None
        path = Path(str(selected)).resolve()
        self.set_active_config(path)
        ui.message("Academic Pipeline — projeto ativo", f"Projeto ativo atualizado:\n{path}")
        return path

    def run_wizard(self) -> int:
        before = {str(path) for path in self.list_activity_configs()}
        command = [
            sys.executable,
            str(TOML_WIZARD),
            "--profile",
            "atividade_local_fgv",
            "--tui-theme",
            "fgv",
        ]
        try:
            rc = subprocess.run(command, cwd=str(ROOT), check=False).returncode
        except Exception as exc:
            ui.message("Academic Pipeline — assistente", f"Não foi possível abrir o assistente:\n{exc}")
            return 1
        after = self.list_activity_configs()
        new_items = [path for path in after if str(path) not in before]
        target = new_items[0] if new_items else (after[0] if after else None)
        if rc == 0 and target:
            self.set_active_config(target)
            ui.message("Academic Pipeline — TOML ativo", f"O assistente foi concluído. TOML ativo:\n{target}")
        elif rc != 0:
            ui.message("Academic Pipeline — assistente", "O assistente foi encerrado ou cancelado antes de gravar um TOML.")
        return int(rc)

    def menu_project(self) -> None:
        while True:
            choice = ui.menu(
                "Academic Pipeline — projeto e configuração",
                self.status_text(),
                [
                    ("new", "[N] Criar nova atividade no assistente FGV", ["n", "1"]),
                    ("select", "[S] Selecionar TOML de atividade existente", ["s", "2"]),
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
                if ui.confirm("Academic Pipeline — projeto ativo", "Remover a referência ao TOML ativo? O arquivo não será excluído.", default=False):
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
        if not ui.confirm(
            "Academic Pipeline — confirmar geração",
            "A geração completa consultará a IA e produzirá document.json, ORG, PDF/DOCX conforme o TOML, além dos relatórios de controle. Confirmar agora?",
            default=False,
        ):
            return False
        rc = self.run_command(["--config", str(path)], label="generate_activity", title="Academic Pipeline — geração completa")
        return rc == 0

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

    def run_guided_flow(self) -> None:
        if self.active_config() is None:
            action = ui.select_one(
                "Academic Pipeline — etapa 1/5",
                "Ainda não há atividade ativa. Escolha como iniciar o fluxo.",
                [
                    ("new", "[N] Criar nova atividade no assistente FGV"),
                    ("select", "[S] Selecionar TOML existente"),
                    ("cancel", "[C] Cancelar"),
                ],
                default="new",
            )
            if action == "new":
                self.run_wizard()
            elif action == "select":
                self.choose_existing_config()
            else:
                return
        path = self.active_config()
        if not path:
            return

        inputs_ok = self.show_inputs()
        if not inputs_ok:
            if ui.confirm("Academic Pipeline — etapa 2/5", "O corpus está pendente. Abrir o assistente para ajustar a configuração?", default=True):
                self.run_wizard()
            return
        if not ui.confirm("Academic Pipeline — etapa 2/5", "Insumos conferidos. Avançar para a validação preventiva?", default=True):
            return

        if not self.validate_config():
            ui.message("Academic Pipeline — etapa 3/5", "A validação encontrou pendências. Corrija o TOML antes de gerar a atividade.")
            return
        if not ui.confirm("Academic Pipeline — etapa 3/5", "Validação concluída. Avançar para a geração completa?", default=True):
            return

        if not self.generate_full():
            return
        output_dir, files = self.list_output_files()
        names = "\n".join(f"✓ {file.name}" for file in files) if files else "Nenhum artefato foi localizado no diretório esperado. Consulte o log."
        ui.message(
            "Academic Pipeline — etapa 5/5",
            f"Fluxo concluído.\n\nDiretório de saída:\n{output_dir}\n\nArtefatos:\n{names}",
            width=126,
        )
        if output_dir and ui.confirm("Academic Pipeline — revisar", "Abrir a pasta de saída agora?", default=False):
            self.open_path(output_dir)

    def main_menu(self) -> int:
        while True:
            choice = ui.menu(
                "Academic Pipeline — Central Operacional FGV",
                self.status_text(),
                [
                    ("guided", "[1] Fluxo guiado: configurar → conferir → validar → gerar → revisar", ["1", "f"]),
                    ("project", "[2] Criar, selecionar ou abrir atividade", ["2", "p"]),
                    ("inputs", "[3] Conferir corpus, orientações e prompts", ["3", "i"]),
                    ("validation", "[4] Validar TOML, ver diretivas e gerar prompt lock", ["4", "v"]),
                    ("generation", "[5] Produzir, renderizar ou recompilar", ["5", "g"]),
                    ("outputs", "[6] Abrir saídas, relatórios e logs", ["6", "o"]),
                    ("exit", "[0] Sair", ["0", "q"]),
                ],
                width=126,
            )
            if choice in {None, "exit"}:
                return 0
            if choice == "guided":
                self.run_guided_flow()
            elif choice == "project":
                self.menu_project()
            elif choice == "inputs":
                self.show_inputs()
            elif choice == "validation":
                action = ui.menu(
                    "Academic Pipeline — validação e diretivas",
                    self.status_text(),
                    [
                        ("check", "[V] Validar configuração", ["v", "1"]),
                        ("prompts", "[P] Ver prompts e diretivas ativos", ["p", "2"]),
                        ("lock", "[L] Gerar prompt lock", ["l", "3"]),
                        ("back", "[0] Voltar", ["0", "q"]),
                    ],
                )
                if action == "check":
                    self.validate_config()
                elif action == "prompts":
                    self.show_prompts()
                elif action == "lock":
                    self.write_prompt_lock()
            elif choice == "generation":
                action = ui.menu(
                    "Academic Pipeline — produção",
                    "Escolha a operação. A geração completa usa IA; as outras opções reutilizam artefatos existentes.",
                    [
                        ("full", "[G] Gerar atividade completa", ["g", "1"]),
                        ("render", "[R] Somente renderizar document.json existente", ["r", "2"]),
                        ("recompile", "[O] Recompilar arquivo ORG", ["o", "3"]),
                        ("back", "[0] Voltar", ["0", "q"]),
                    ],
                )
                if action == "full":
                    self.generate_full()
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
