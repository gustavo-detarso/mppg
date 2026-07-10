#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path

PATCH_NAME = "tui_separar_contextos_v1"

HELPERS_AND_STATUS = r'''
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
'''

NEW_MENU_PRISMA_ARTICLE_FLOW = r'''
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
'''


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        target = base / "app_bundle" / "scripts" / "pipeline" / "academic_pipeline_tui.py"
        if target.exists():
            return base
    raise SystemExit("ERRO: rode este aplicador na raiz do projeto academic_pipeline_rc10_7_conformidade.")


def backup(path: Path) -> Path:
    bak = path.with_name(path.name + f".bak_{PATCH_NAME}_{stamp()}")
    shutil.copy2(path, bak)
    print(f"[OK] Backup: {bak}")
    return bak


def replace_region(txt: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = txt.find(start_marker)
    if start < 0:
        raise SystemExit(f"ERRO: não localizei o início da região {label}: {start_marker!r}")
    end = txt.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"ERRO: não localizei o fim da região {label}: {end_marker!r}")
    return txt[:start] + replacement.rstrip() + "\n\n" + txt[end:]


def patch_status(txt: str) -> str:
    # Remove helpers/status de execução anterior, se houver.
    if "    def _global_project_context_text(self) -> str:" in txt:
        txt = replace_region(
            txt,
            "    def _global_project_context_text(self) -> str:\n",
            "    def require_config(self) -> Path | None:\n",
            HELPERS_AND_STATUS,
            "helpers/status já instalados",
        )
    else:
        txt = replace_region(
            txt,
            "    def status_text(self) -> str:\n",
            "    def require_config(self) -> Path | None:\n",
            HELPERS_AND_STATUS,
            "status_text",
        )
    return txt


def patch_prisma_menu(txt: str) -> str:
    txt = replace_region(
        txt,
        "    def menu_prisma_article_flow(self) -> None:\n",
        "    # ------------------------------------------------------------------\n    # Saídas e integração com desktop\n",
        NEW_MENU_PRISMA_ARTICLE_FLOW,
        "menu_prisma_article_flow",
    )
    return txt


def main() -> int:
    parser = argparse.ArgumentParser(description="Separa visualmente o projeto global da pipeline do artigo PRISMA ativo na TUI.")
    parser.add_argument("--skip-tests", action="store_true", help="Aplica sem rodar py_compile e verificações textuais.")
    args = parser.parse_args()

    root = find_root()
    tui = root / "app_bundle" / "scripts" / "pipeline" / "academic_pipeline_tui.py"
    backup(tui)

    txt = tui.read_text(encoding="utf-8", errors="ignore")
    if "def menu_prisma_article_flow" not in txt:
        raise SystemExit("ERRO: fluxo Artigo PRISMA não encontrado. Aplique antes os patches de fluxo/robustez.")

    txt = patch_status(txt)
    txt = patch_prisma_menu(txt)
    txt = txt.replace("Projeto ativo: nenhum. Selecione “Novo documento — fluxo completo”.", "Projeto global selecionado: nenhum.")
    txt = txt.replace("Projeto ativo atualizado", "TOML global da pipeline atualizado")
    txt = txt.replace("Academic Pipeline — projeto ativo", "Academic Pipeline — TOML global da pipeline")

    tui.write_text(txt, encoding="utf-8")

    if not args.skip_tests:
        py_compile.compile(str(tui), doraise=True)
        final = tui.read_text(encoding="utf-8", errors="ignore")
        required = [
            "Contexto global da pipeline:",
            "Contexto do artigo PRISMA:",
            "Projeto global selecionado:",
            "TOML PRISMA preliminar/global selecionado:",
            "TOML do artigo final:",
        ]
        missing = [token for token in required if token not in final]
        if missing:
            raise SystemExit("ERRO: verificações textuais falharam. Ausentes: " + ", ".join(missing))
        if "Projeto ativo:" in final:
            raise SystemExit("ERRO: ainda existe 'Projeto ativo:' no arquivo da TUI.")

    print("\n[OK] Patch aplicado: separação visual entre projeto global e artigo PRISMA ativo.")
    print("\nAgora a tela principal exibirá dois blocos separados:")
    print("  - Contexto global da pipeline")
    print("  - Contexto do artigo PRISMA")
    print("\nAbra a TUI com:")
    print("pipenv run python app_bundle/scripts/pipeline/academic_pipeline_tui.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
