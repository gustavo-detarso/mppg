#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TUI operacional para o academic_pipeline.

Centraliza o fluxo de produção de atividades acadêmicas no terminal:
configurar -> conferir insumos -> validar -> gerar -> revisar artefatos.

Não substitui o pipeline nem o gerador TOML. Atua como camada operacional que
chama os comandos canônicos e preserva todos os artefatos/auditorias gerados.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve()
APP = HERE.parents[2]
ROOT = APP.parent
PIPELINE = HERE.with_name("academic_pipeline_rc10.py")
TOML_WIZARD = HERE.with_name("academic_pipeline_toml_generator_interativo.py")
STATE_PATH = APP / ".academic_pipeline_tui_state.json"
LOG_DIR = APP / "output" / "tui_logs"


class TerminalUI:
    """Camada de terminal sem dependências externas."""

    def __init__(self, *, no_clear: bool = False) -> None:
        env_no_clear = os.getenv("ACADEMIC_PIPELINE_TUI_NO_CLEAR", "").strip().lower()
        self.no_clear = bool(no_clear) or env_no_clear in {"1", "true", "s", "sim", "yes", "y"}
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # Estado e terminal
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

    def clear(self) -> None:
        if self.no_clear:
            print("\n" + "=" * 94 + "\n")
            return
        os.system("cls" if os.name == "nt" else "clear")

    def pause(self, prompt: str = "Pressione Enter para continuar") -> None:
        try:
            input(f"\n{prompt}...")
        except EOFError:
            return

    def header(self, title: str, *, subtitle: str = "") -> None:
        self.clear()
        print("╔" + "═" * 92 + "╗")
        print("║ Academic Pipeline — Central Operacional de Atividades".ljust(93) + "║")
        print("║ Fluxo: configurar → conferir → validar → gerar → revisar".ljust(93) + "║")
        print("╚" + "═" * 92 + "╝")
        if subtitle:
            print(subtitle)
        print(f"\n{title}\n" + "─" * 94)

    @staticmethod
    def ask(prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        value = input(f"{prompt}{suffix}: ").strip()
        return value or default

    @staticmethod
    def yes_no(prompt: str, default: bool = False) -> bool:
        mark = "s" if default else "n"
        while True:
            raw = input(f"{prompt} (s/n) [{mark}]: ").strip().lower()
            if not raw:
                return default
            if raw in {"s", "sim", "y", "yes"}:
                return True
            if raw in {"n", "nao", "não", "no"}:
                return False
            print("Responda s ou n.")

    # ------------------------------------------------------------------
    # TOML, caminhos e status
    # ------------------------------------------------------------------

    @staticmethod
    def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
        value = cfg.get(name, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def load_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def resolve_config_path(config_path: Path, raw: Any) -> Path | None:
        value = str(raw or "").strip()
        if not value or value.startswith("profile://"):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return path.resolve()

    @staticmethod
    def is_activity_config(path: Path) -> bool:
        try:
            cfg = TerminalUI.load_toml(path)
        except Exception:
            return False
        project = TerminalUI._section(cfg, "projeto")
        document = TerminalUI._section(cfg, "documento")
        preset = str(project.get("preset") or "").strip().lower()
        doc_type = str(document.get("tipo_documento") or "").strip().lower()
        return preset == "atividade_local_fgv" or doc_type == "atividade"

    def list_activity_configs(self) -> list[Path]:
        base = APP / "projetos"
        if not base.exists():
            return []
        items = [p.resolve() for p in base.rglob("*.toml") if p.is_file() and self.is_activity_config(p)]
        return sorted(items, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)

    def cfg_output_dir(self, config_path: Path, cfg: dict[str, Any] | None = None) -> Path:
        cfg = cfg if cfg is not None else self.load_toml(config_path)
        paths = self._section(cfg, "paths")
        project = self._section(cfg, "projeto")
        raw_base = paths.get("document_output_dir") or "../../output/documento"
        base = self.resolve_config_path(config_path, raw_base) or (config_path.parent / "output" / "documento")
        prefix = str(paths.get("document_prefix") or project.get("nome") or "documento").strip() or "documento"
        return base / prefix if bool(paths.get("create_document_subdir", True)) else base

    def cfg_prefix(self, cfg: dict[str, Any]) -> str:
        paths = self._section(cfg, "paths")
        project = self._section(cfg, "projeto")
        return str(paths.get("document_prefix") or project.get("nome") or "documento").strip() or "documento"

    def input_rows(self, config_path: Path, cfg: dict[str, Any] | None = None) -> list[tuple[str, str, str]]:
        cfg = cfg if cfg is not None else self.load_toml(config_path)
        local = self._section(cfg, "documentos_locais")
        orientations = self._section(cfg, "orientacoes")
        prompts = self._section(cfg, "prompts")
        rows: list[tuple[str, str, str]] = []

        def append_path(label: str, raw: Any, *, required: bool = False) -> None:
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
        append_path("Corpus ZIP", input_zip, required=not input_dir)
        append_path("Corpus pasta", input_dir, required=not input_zip)

        orient_paths = orientations.get("paths", [])
        if isinstance(orient_paths, list) and orient_paths:
            for index, raw in enumerate(orient_paths, start=1):
                append_path(f"Orientação {index}", raw)
        else:
            inline = str(orientations.get("inline") or "").strip()
            rows.append(("Orientações", "texto no TOML" if inline else "—", "OK" if inline else "não informado"))

        activity = self._section(cfg, "atividade")
        source_mode = str(activity.get("fonte_dados_atividade") or "").strip()
        ai_mode = bool(activity.get("gerar_dados_atividade_ia", False))
        source_status = "IA" if ai_mode or source_mode == "ia" else (source_mode or "manual")
        rows.append(("Dados da atividade", source_status, "OK"))

        for key in ("global_paths", "institution_paths", "atividade_paths", "document_paths"):
            values = prompts.get(key, [])
            if not isinstance(values, list):
                continue
            for index, raw in enumerate(values, start=1):
                append_path(f"Prompt {key} {index}", raw)

        doi = local.get("doi_manifest_path")
        if str(doi or "").strip():
            append_path("Manifesto DOI", doi)
        return rows

    def activity_status(self) -> dict[str, Any]:
        config_path = self.active_config()
        result: dict[str, Any] = {
            "config": config_path,
            "config_ok": False,
            "inputs_ok": False,
            "outputs": [],
            "output_dir": None,
            "title": "",
            "project": "",
        }
        if not config_path:
            return result
        try:
            cfg = self.load_toml(config_path)
        except Exception:
            return result
        result["config_ok"] = True
        result["title"] = str(self._section(cfg, "documento").get("titulo_trabalho") or self._section(cfg, "atividade").get("titulo_trabalho") or "")
        result["project"] = str(self._section(cfg, "projeto").get("nome") or config_path.parent.name)
        rows = self.input_rows(config_path, cfg)
        critical = [status for label, _path, status in rows if label in {"Corpus ZIP", "Corpus pasta"} and status in {"PENDENTE", "AUSENTE"}]
        result["inputs_ok"] = not critical
        out_dir = self.cfg_output_dir(config_path, cfg)
        result["output_dir"] = out_dir
        if out_dir.exists():
            result["outputs"] = sorted([p for p in out_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
        return result

    def print_status_panel(self) -> None:
        status = self.activity_status()
        config = status["config"]
        print("Projeto ativo")
        print(f"  Projeto: {status['project'] or 'nenhum selecionado'}")
        print(f"  Título: {status['title'] or '—'}")
        print(f"  TOML: {config or '—'}")
        print(f"  1. Configuração: {'✓ pronta' if status['config_ok'] else '○ pendente'}")
        print(f"  2. Insumos: {'✓ localizados' if status['inputs_ok'] else '○ conferir corpus'}")
        print(f"  3. Saídas: {'✓ ' + str(len(status['outputs'])) + ' arquivo(s)' if status['outputs'] else '○ ainda não geradas'}")
        print()

    # ------------------------------------------------------------------
    # Execução de subprocessos
    # ------------------------------------------------------------------

    def run_command(self, args: list[str], *, label: str, log_output: bool = True) -> int:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{timestamp}_{label}.log"
        command = [sys.executable, str(PIPELINE), *args]
        print("\nComando operacional:")
        print("  " + " ".join(command))
        print("\nSaída do processo:\n" + "─" * 94)
        if not log_output:
            return subprocess.run(command, cwd=ROOT).returncode
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("Comando:\n" + " ".join(command) + "\n\nSaída:\n")
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = proc.wait()
        except KeyboardInterrupt:
            print("\nInterrupção solicitada. Aguardando o processo finalizar...")
            returncode = 130
        print("─" * 94)
        print(f"Log operacional: {log_path}")
        print("Resultado: " + ("SUCESSO" if returncode == 0 else f"FALHOU (código {returncode})"))
        return returncode

    def run_wizard(self) -> int:
        before = {p.resolve(): p.stat().st_mtime_ns for p in self.list_activity_configs()}
        args = [sys.executable, str(TOML_WIZARD), "--profile", "atividade_local_fgv"]
        if self.no_clear:
            args.append("--no-clear")
        print("\nO assistente abaixo cria ou reconfigura o TOML da atividade.")
        print("Ao concluir, esta TUI tentará selecionar automaticamente o TOML criado ou alterado.\n")
        returncode = subprocess.run(args, cwd=ROOT).returncode
        after = self.list_activity_configs()
        new_or_changed = [
            p for p in after
            if p not in before or p.stat().st_mtime_ns > before[p]
        ]
        selected = new_or_changed[0] if new_or_changed else (after[0] if after else None)
        if returncode == 0 and selected:
            self.set_active_config(selected)
            print(f"\nProjeto ativo atualizado: {selected}")
        elif returncode == 0:
            print("\nO TOML foi gerado, mas não consegui identificá-lo automaticamente. Selecione-o no menu de projetos.")
        return returncode

    # ------------------------------------------------------------------
    # Menus operacionais
    # ------------------------------------------------------------------

    def require_config(self) -> Path | None:
        path = self.active_config()
        if path:
            return path
        print("Nenhuma atividade está selecionada. Use 'Projeto e configuração' primeiro.")
        self.pause()
        return None

    def menu_project(self) -> None:
        while True:
            self.header("Projeto e configuração")
            self.print_status_panel()
            print("1. Criar nova atividade ou reconfigurar via assistente")
            print("2. Selecionar uma atividade existente")
            print("3. Exibir resumo do TOML ativo")
            print("4. Abrir TOML ativo no editor configurado")
            print("0. Voltar")
            choice = self.ask("Opção", "0")
            if choice == "1":
                self.run_wizard()
                self.pause()
            elif choice == "2":
                self.select_existing_config()
            elif choice == "3":
                self.show_config_summary()
            elif choice == "4":
                self.edit_active_config()
            elif choice == "0":
                return
            else:
                print("Opção inválida.")
                self.pause()

    def select_existing_config(self) -> None:
        configs = self.list_activity_configs()
        self.header("Selecionar atividade existente")
        if not configs:
            print("Nenhum TOML de atividade encontrado em app_bundle/projetos.")
            self.pause()
            return
        for index, path in enumerate(configs, start=1):
            try:
                cfg = self.load_toml(path)
                title = str(self._section(cfg, "documento").get("titulo_trabalho") or self._section(cfg, "atividade").get("titulo_trabalho") or "sem título")
                project = str(self._section(cfg, "projeto").get("nome") or path.parent.name)
                print(f"{index}. {project} — {title}")
                print(f"   {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            except Exception:
                print(f"{index}. {path}")
        print("0. Cancelar")
        raw = self.ask("Escolha", "0")
        if raw.isdigit() and 1 <= int(raw) <= len(configs):
            self.set_active_config(configs[int(raw) - 1])
            print("Atividade selecionada.")
        elif raw != "0":
            print("Escolha inválida.")
        self.pause()

    def show_config_summary(self) -> None:
        path = self.require_config()
        if not path:
            return
        self.header("Resumo da configuração", subtitle=str(path))
        try:
            cfg = self.load_toml(path)
        except Exception as exc:
            print(f"Não foi possível ler o TOML: {exc}")
            self.pause()
            return
        project = self._section(cfg, "projeto")
        activity = self._section(cfg, "atividade")
        research = self._section(cfg, "pesquisa")
        document = self._section(cfg, "documento")
        print("Metadados")
        for label, value in [
            ("Projeto", project.get("nome")),
            ("Título", document.get("titulo_trabalho") or activity.get("titulo_trabalho")),
            ("Disciplina", activity.get("disciplina") or document.get("discipline_name")),
            ("Professor", activity.get("professor") or document.get("professor_name")),
            ("Fonte dos dados", activity.get("fonte_dados_atividade")),
            ("Dados inferidos por IA", activity.get("gerar_dados_atividade_ia")),
            ("Tema", research.get("tema")),
            ("Pergunta", research.get("pergunta_pesquisa")),
        ]:
            text = str(value or "—")
            print(f"  {label}: {textwrap.shorten(text, width=118, placeholder='…')}")
        print("\nSaídas configuradas")
        print(f"  Diretório: {self.cfg_output_dir(path, cfg)}")
        print(f"  ORG: {bool(document.get('exportar_org', True))}")
        print(f"  PDF: {bool(document.get('exportar_pdf', True))}")
        print(f"  DOCX: {bool(document.get('exportar_docx', True))}")
        self.pause()

    def edit_active_config(self) -> None:
        path = self.require_config()
        if not path:
            return
        editor = os.getenv("EDITOR") or os.getenv("VISUAL") or "emacs"
        self.header("Editar TOML", subtitle=f"Editor: {editor}")
        print(f"Abrindo: {path}")
        try:
            subprocess.run([editor, str(path)], cwd=ROOT, check=False)
        except FileNotFoundError:
            print(f"Editor '{editor}' não encontrado. Defina EDITOR ou abra manualmente o arquivo indicado.")
        self.pause()

    def menu_inputs(self) -> None:
        path = self.require_config()
        if not path:
            return
        self.header("Conferência de corpus, orientações e prompts", subtitle=str(path))
        try:
            cfg = self.load_toml(path)
            rows = self.input_rows(path, cfg)
        except Exception as exc:
            print(f"Não foi possível ler o TOML: {exc}")
            self.pause()
            return
        for label, item_path, status in rows:
            marker = "✓" if status in {"OK", "IA", "interno do perfil"} else ("!" if status in {"AUSENTE", "PENDENTE"} else "·")
            print(f"{marker} {label}: {status}")
            print(f"    {item_path}")
        print("\nObservação: dados da atividade no modo IA são inferidos somente durante a geração, a partir do corpus, orientações e prompts ativos.")
        self.pause()

    def menu_validation(self) -> None:
        path = self.require_config()
        if not path:
            return
        while True:
            self.header("Validação e diretivas ativas", subtitle=str(path))
            print("1. Validar TOML e caminhos")
            print("2. Mostrar prompts e diretivas efetivas")
            print("3. Gerar prompt_lock sem gerar o trabalho")
            print("4. Executar diagnóstico do ambiente")
            print("0. Voltar")
            choice = self.ask("Opção", "0")
            if choice == "1":
                self.run_command(["--config", str(path), "--check-config"], label="check_config")
                self.pause()
            elif choice == "2":
                self.run_command(["--config", str(path), "--show-prompts"], label="show_prompts")
                self.pause()
            elif choice == "3":
                self.run_command(["--config", str(path), "--write-prompt-lock"], label="prompt_lock")
                self.pause()
            elif choice == "4":
                self.run_command(["--config", str(path), "--doctor"], label="doctor")
                self.pause()
            elif choice == "0":
                return
            else:
                print("Opção inválida.")
                self.pause()

    def validate_current(self, path: Path) -> bool:
        self.header("Validação preventiva", subtitle=str(path))
        rc = self.run_command(["--config", str(path), "--check-config"], label="check_config")
        if rc != 0:
            print("\nA geração foi bloqueada. Corrija o TOML ou os insumos antes de continuar.")
            self.pause()
            return False
        return True

    def generate_full(self) -> bool:
        path = self.require_config()
        if not path:
            return False
        if not self.validate_current(path):
            return False
        self.header("Geração completa da atividade", subtitle=str(path))
        print("A execução irá ler o corpus, montar bibliografia, chamar a IA, gerar document.json, ORG, PDF/DOCX e relatórios de qualidade/conformidade.")
        if not self.yes_no("A geração pode consumir créditos da API. Confirmar execução", False):
            print("Geração cancelada antes de iniciar.")
            self.pause()
            return False
        rc = self.run_command(["--config", str(path)], label="generate_activity")
        if rc == 0:
            print("\nAtividade gerada com sucesso. Consulte 'Saídas e relatórios' para conferir os artefatos.")
            self.pause()
            return True
        print("\nA geração falhou. Consulte o log indicado acima e o relatório de configuração.")
        self.pause()
        return False

    def rerender_existing(self) -> None:
        path = self.require_config()
        if not path:
            return
        try:
            cfg = self.load_toml(path)
            out_dir = self.cfg_output_dir(path, cfg)
            prefix = self.cfg_prefix(cfg)
        except Exception as exc:
            print(f"Não foi possível ler o TOML: {exc}")
            self.pause()
            return
        default_json = out_dir / f"{prefix}.document.json"
        self.header("Somente renderizar", subtitle=str(path))
        raw = self.ask("Caminho do document.json", str(default_json) if default_json.exists() else "")
        doc_json = Path(raw).expanduser()
        if not doc_json.exists():
            print("document.json não encontrado.")
            self.pause()
            return
        if not self.yes_no("Recompilar ORG/PDF/DOCX sem chamar a IA", True):
            return
        self.run_command(["--config", str(path), "--somente-renderizar", "--document-json", str(doc_json)], label="rerender")
        self.pause()

    def recompile_org(self) -> None:
        path = self.require_config()
        if not path:
            return
        try:
            cfg = self.load_toml(path)
            out_dir = self.cfg_output_dir(path, cfg)
            prefix = self.cfg_prefix(cfg)
        except Exception as exc:
            print(f"Não foi possível ler o TOML: {exc}")
            self.pause()
            return
        default_org = out_dir / f"{prefix}.org"
        self.header("Recompilar ORG", subtitle=str(path))
        raw = self.ask("Caminho do arquivo ORG", str(default_org) if default_org.exists() else "")
        org = Path(raw).expanduser()
        if not org.exists():
            print("Arquivo ORG não encontrado.")
            self.pause()
            return
        self.run_command(["--config", str(path), "--recompile", "--org", str(org)], label="recompile_org")
        self.pause()

    def menu_generation(self) -> None:
        while True:
            self.header("Produção da atividade")
            self.print_status_panel()
            print("1. Gerar atividade completa (valida antes; usa IA)")
            print("2. Somente renderizar document.json existente (sem IA)")
            print("3. Recompilar um ORG existente (sem IA)")
            print("0. Voltar")
            choice = self.ask("Opção", "0")
            if choice == "1":
                self.generate_full()
            elif choice == "2":
                self.rerender_existing()
            elif choice == "3":
                self.recompile_org()
            elif choice == "0":
                return
            else:
                print("Opção inválida.")
                self.pause()

    def menu_outputs(self) -> None:
        path = self.require_config()
        if not path:
            return
        try:
            cfg = self.load_toml(path)
            out_dir = self.cfg_output_dir(path, cfg)
        except Exception as exc:
            print(f"Não foi possível ler o TOML: {exc}")
            self.pause()
            return
        self.header("Saídas, relatórios e logs", subtitle=str(out_dir))
        print("Artefatos do projeto:")
        files = sorted([p for p in out_dir.glob("*") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True) if out_dir.exists() else []
        if files:
            for index, file in enumerate(files, start=1):
                stamp = datetime.fromtimestamp(file.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
                print(f"{index:>2}. {file.name} — {file.stat().st_size / 1024:.1f} KiB — {stamp}")
        else:
            print("  Nenhuma saída gerada ainda.")
        print(f"\nLogs da TUI: {LOG_DIR}")
        print("\n1. Abrir diretório de saída no gerenciador de arquivos")
        print("2. Abrir um artefato com o aplicativo padrão")
        print("3. Mostrar o relatório de qualidade no terminal")
        print("4. Mostrar o relatório de conformidade no terminal")
        print("0. Voltar")
        choice = self.ask("Opção", "0")
        if choice == "1":
            self.open_path(out_dir)
        elif choice == "2":
            self.open_artifact(files)
        elif choice == "3":
            self.show_report_file(files, "quality_report")
        elif choice == "4":
            self.show_report_file(files, "compliance")
        elif choice != "0":
            print("Opção inválida.")
        self.pause()

    def open_path(self, path: Path) -> None:
        if not path.exists():
            print("O diretório ainda não existe.")
            return
        opener = shutil.which("xdg-open")
        if not opener:
            print(f"Abra manualmente: {path}")
            return
        subprocess.Popen([opener, str(path)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Solicitado ao sistema abrir: {path}")

    def open_artifact(self, files: list[Path]) -> None:
        if not files:
            print("Nenhum artefato disponível.")
            return
        for index, file in enumerate(files, start=1):
            print(f"{index}. {file.name}")
        raw = self.ask("Arquivo", "0")
        if not raw.isdigit() or not (1 <= int(raw) <= len(files)):
            return
        self.open_path(files[int(raw) - 1])

    def show_report_file(self, files: list[Path], token: str) -> None:
        matches = [p for p in files if token.lower() in p.name.lower() and p.suffix.lower() in {".md", ".txt"}]
        if not matches:
            print("Relatório não encontrado.")
            return
        file = matches[0]
        print(f"\n{file.name}\n" + "─" * 94)
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
            print(content[:20000])
            if len(content) > 20000:
                print("\n[Relatório truncado no terminal; abra o arquivo para leitura integral.]")
        except Exception as exc:
            print(f"Não foi possível ler o relatório: {exc}")

    def run_guided_flow(self) -> None:
        """Fluxo obrigatório em etapas, equivalente a um procedimento operacional."""
        self.header("Fluxo guiado — atividade acadêmica")
        print("Este fluxo não pressupõe que você memorize comandos. Cada etapa só avança quando a anterior estiver em condição operacional.\n")
        path = self.active_config()
        if not path:
            print("ETAPA 1/5 — Criar a configuração da atividade")
            if not self.yes_no("Abrir agora o assistente de configuração", True):
                return
            rc = self.run_wizard()
            if rc != 0 or not self.active_config():
                print("Não foi possível preparar um TOML ativo. Fluxo interrompido.")
                self.pause()
                return
            path = self.active_config()

        assert path is not None
        self.header("Fluxo guiado — conferência dos insumos", subtitle=str(path))
        print("ETAPA 2/5 — Corpus, orientações e prompts")
        try:
            cfg = self.load_toml(path)
            rows = self.input_rows(path, cfg)
        except Exception as exc:
            print(f"TOML inválido ou inacessível: {exc}")
            self.pause()
            return
        for label, item_path, status in rows:
            marker = "✓" if status in {"OK", "IA", "interno do perfil"} else ("!" if status in {"AUSENTE", "PENDENTE"} else "·")
            print(f"{marker} {label}: {status} — {item_path}")
        corpus_missing = any(label in {"Corpus ZIP", "Corpus pasta"} and status in {"AUSENTE", "PENDENTE"} for label, _item, status in rows)
        if corpus_missing:
            print("\nO corpus não está disponível. Reabra o assistente e corrija o caminho antes de gerar.")
            if self.yes_no("Abrir o assistente de configuração agora", True):
                self.run_wizard()
            self.pause()
            return
        if not self.yes_no("Insumos conferidos. Avançar para a validação", True):
            return

        self.header("Fluxo guiado — validação", subtitle=str(path))
        print("ETAPA 3/5 — Validação preventiva do TOML")
        rc = self.run_command(["--config", str(path), "--check-config"], label="guided_check_config")
        if rc != 0:
            print("\nA validação identificou pendências. Corrija-as antes de continuar.")
            self.pause()
            return
        if not self.yes_no("Validação concluída. Avançar para a geração", True):
            return

        self.header("Fluxo guiado — geração", subtitle=str(path))
        print("ETAPA 4/5 — Geração completa")
        print("Será feita uma chamada à IA e serão produzidos document.json, ORG, PDF/DOCX conforme o TOML e relatórios operacionais.")
        if not self.yes_no("Confirmar geração agora", False):
            print("Fluxo encerrado antes da chamada à IA.")
            self.pause()
            return
        rc = self.run_command(["--config", str(path)], label="guided_generate_activity")
        if rc != 0:
            print("\nGeração não concluída. Verifique o log operacional mostrado acima.")
            self.pause()
            return

        self.header("Fluxo guiado — revisão final", subtitle=str(path))
        print("ETAPA 5/5 — Conferir os artefatos produzidos")
        try:
            cfg = self.load_toml(path)
            out_dir = self.cfg_output_dir(path, cfg)
            files = sorted([p for p in out_dir.glob("*") if p.is_file()], key=lambda p: p.name) if out_dir.exists() else []
        except Exception:
            files = []
            out_dir = None
        print(f"Diretório de saída: {out_dir or 'não identificado'}")
        if files:
            for file in files:
                print(f"✓ {file.name}")
        else:
            print("Nenhum artefato foi encontrado no diretório esperado. Consulte o log da execução.")
        if out_dir and self.yes_no("Abrir diretório de saída", False):
            self.open_path(out_dir)
        self.pause("Fluxo concluído. Pressione Enter para retornar ao painel")

    def main_menu(self) -> int:
        while True:
            self.header("Painel operacional")
            self.print_status_panel()
            print("1. Fluxo guiado: configurar → conferir → validar → gerar → revisar")
            print("2. Projeto e configuração")
            print("3. Conferir corpus, orientações e prompts")
            print("4. Validação e diretivas ativas")
            print("5. Produção da atividade")
            print("6. Saídas, relatórios e logs")
            print("0. Sair")
            choice = self.ask("Opção", "1")
            if choice == "1":
                self.run_guided_flow()
            elif choice == "2":
                self.menu_project()
            elif choice == "3":
                self.menu_inputs()
            elif choice == "4":
                self.menu_validation()
            elif choice == "5":
                self.menu_generation()
            elif choice == "6":
                self.menu_outputs()
            elif choice == "0":
                self.clear()
                print("TUI encerrada.")
                return 0
            else:
                print("Opção inválida.")
                self.pause()


def run_tui(*, no_clear: bool = False) -> int:
    """Ponto de integração utilizado por academic_pipeline_rc10.py --tui."""
    return TerminalUI(no_clear=no_clear).main_menu()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TUI operacional do academic_pipeline")
    parser.add_argument("--no-clear", action="store_true", help="Não limpa a tela entre menus")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_tui(no_clear=bool(args.no_clear))


if __name__ == "__main__":
    raise SystemExit(main())
