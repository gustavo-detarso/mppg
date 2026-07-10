#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import py_compile

NEW_MAIN_MENU = '\n    def main_menu(self) -> int:\n        """Menu principal reorganizado por categorias de uso.\n\n        A intenção é separar claramente:\n        - artigo PRISMA completo, que é um fluxo metodológico próprio;\n        - documentos acadêmicos gerais, que usam o assistente tradicional;\n        - administração de TOML/insumos;\n        - validação, produção técnica e saídas.\n        """\n        while True:\n            description = (\n                self.status_text()\n                + "\\n\\n"\n                + "Escolha a categoria do trabalho.\\n"\n                + "Para artigo com revisão estruturada/PRISMA, use a primeira opção. "\n                + "Para atividades, resumos, papers locais e dissertações, use documentos acadêmicos gerais."\n            )\n            choice = ui.menu(\n                "Academic Pipeline — Central Operacional FGV",\n                description,\n                [\n                    ("prisma_article", "[1] Artigos PRISMA e revisão estruturada — briefing → PRISMA → PDF final", ["1", "a"]),\n                    ("general_documents", "[2] Documentos acadêmicos gerais — atividade, resumo, paper, dissertação", ["2", "d"]),\n                    ("configuration", "[3] Projetos, TOML e insumos — criar, selecionar, abrir e conferir", ["3", "p"]),\n                    ("validation", "[4] Validação e prompts — checar TOML, diretivas e prompt lock", ["4", "v"]),\n                    ("production", "[5] Produção técnica — gerar, importar triagem, renderizar ou recompilar", ["5", "g"]),\n                    ("outputs", "[6] Saídas e relatórios — abrir PDF, DOCX, ORG, logs e pastas", ["6", "o"]),\n                    ("exit", "[0] Sair", ["0", "q"]),\n                ],\n                width=126,\n            )\n            if choice in {None, "exit"}:\n                return 0\n\n            if choice == "prisma_article":\n                self.menu_prisma_article_flow()\n\n            elif choice == "general_documents":\n                while True:\n                    action = ui.menu(\n                        "Academic Pipeline — documentos acadêmicos gerais",\n                        (\n                            self.status_text()\n                            + "\\n\\n"\n                            + "Use este grupo para produtos que não exigem o ciclo completo PRISMA→XLSX→full text.\\n"\n                            + "Exemplos: atividade de disciplina, resumo de artigos, paper local, dissertação local ou renderização de document.json."\n                        ),\n                        [\n                            ("new", "[1] Criar novo documento pelo assistente", ["1", "n"]),\n                            ("resume", "[2] Retomar TOML existente: conferir, validar e gerar", ["2", "r"]),\n                            ("profiles", "[3] Ver perfis disponíveis no assistente", ["3", "f"]),\n                            ("back", "[0] Voltar ao menu principal", ["0", "q", "v"]),\n                        ],\n                        width=126,\n                    )\n                    if action in {None, "back"}:\n                        break\n                    if action == "new":\n                        self.run_new_document_flow()\n                    elif action == "resume":\n                        self.run_guided_flow()\n                    elif action == "profiles":\n                        lines = ["Perfis disponíveis no assistente:", ""]\n                        for key, label in DOCUMENT_PROFILES:\n                            clean = re.sub(r"^\\[[0-9]+\\]\\s*", "", label)\n                            lines.append(f"• {key}: {clean}")\n                        ui.message("Academic Pipeline — perfis de documento", "\\n".join(lines), width=126)\n\n            elif choice == "configuration":\n                while True:\n                    action = ui.menu(\n                        "Academic Pipeline — projetos, TOML e insumos",\n                        self.status_text(),\n                        [\n                            ("project", "[1] Criar/selecionar/abrir TOML", ["1", "t"]),\n                            ("inputs", "[2] Conferir corpus, orientações, prompts e caminhos", ["2", "i"]),\n                            ("select", "[3] Selecionar TOML existente diretamente", ["3", "s"]),\n                            ("back", "[0] Voltar ao menu principal", ["0", "q", "v"]),\n                        ],\n                        width=126,\n                    )\n                    if action in {None, "back"}:\n                        break\n                    if action == "project":\n                        self.menu_project()\n                    elif action == "inputs":\n                        self.show_inputs()\n                    elif action == "select":\n                        self.choose_existing_config()\n\n            elif choice == "validation":\n                while True:\n                    action = ui.menu(\n                        "Academic Pipeline — validação e prompts",\n                        self.status_text(),\n                        [\n                            ("check", "[1] Validar configuração/TOML", ["1", "v"]),\n                            ("prompts", "[2] Ver prompts e diretivas ativos", ["2", "p"]),\n                            ("lock", "[3] Gerar prompt lock", ["3", "l"]),\n                            ("back", "[0] Voltar ao menu principal", ["0", "q"]),\n                        ],\n                        width=126,\n                    )\n                    if action in {None, "back"}:\n                        break\n                    if action == "check":\n                        self.validate_config()\n                    elif action == "prompts":\n                        self.show_prompts()\n                    elif action == "lock":\n                        self.write_prompt_lock()\n\n            elif choice == "production":\n                while True:\n                    action = ui.menu(\n                        "Academic Pipeline — produção técnica",\n                        (\n                            "Operações de execução direta.\\n"\n                            "Em fluxos PRISMA completos, prefira a categoria 1, que apresenta as etapas na ordem metodológica."\n                        ),\n                        [\n                            ("full", "[1] Gerar documento completo ou executar busca PRISMA do TOML ativo", ["1", "g"]),\n                            ("triage", "[2] Importar planilha de triagem PRISMA preenchida", ["2", "t"]),\n                            ("render", "[3] Somente renderizar document.json existente", ["3", "r"]),\n                            ("recompile", "[4] Recompilar arquivo ORG", ["4", "o"]),\n                            ("back", "[0] Voltar ao menu principal", ["0", "q"]),\n                        ],\n                        width=126,\n                    )\n                    if action in {None, "back"}:\n                        break\n                    if action == "full":\n                        self.generate_full()\n                    elif action == "triage":\n                        self.import_prisma_triage()\n                    elif action == "render":\n                        self.rerender_existing()\n                    elif action == "recompile":\n                        self.recompile_org()\n\n            elif choice == "outputs":\n                self.menu_outputs()\n'


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "app_bundle" / "scripts" / "pipeline" / "academic_pipeline_tui.py").exists():
            return base
    raise SystemExit("ERRO: rode este aplicador na raiz do projeto academic_pipeline_rc10_7_conformidade.")


def backup(path: Path, tag: str) -> Path:
    bak = path.with_name(path.name + f".bak_{tag}_{stamp()}")
    shutil.copy2(path, bak)
    print(f"[OK] Backup: {bak}")
    return bak


def replace_main_menu(txt: str) -> str:
    marker = "\n\n# Compatibilidade com os testes"
    start = txt.find("    def main_menu(self) -> int:\n")
    if start < 0:
        raise SystemExit("ERRO: não localizei def main_menu(self) em academic_pipeline_tui.py.")
    end = txt.find(marker, start)
    if end < 0:
        raise SystemExit("ERRO: não localizei o marcador de compatibilidade após main_menu.")
    return txt[:start] + NEW_MAIN_MENU.rstrip() + txt[end:]


def main() -> int:
    root = find_root()
    tui = root / "app_bundle" / "scripts" / "pipeline" / "academic_pipeline_tui.py"
    backup(tui, "tui_menu_categorias_v1")

    txt = tui.read_text(encoding="utf-8", errors="ignore")

    if "def menu_prisma_article_flow" not in txt:
        raise SystemExit(
            "ERRO: o fluxo Artigo PRISMA ainda não está instalado.\n"
            "Aplique primeiro o patch_tui_fluxo_artigo_prisma_v1 e depois rode este patch de menu."
        )

    txt = replace_main_menu(txt)
    tui.write_text(txt, encoding="utf-8")
    py_compile.compile(str(tui), doraise=True)

    print(f"[OK] Menu principal reorganizado por categorias: {tui}")
    print("\nNovo menu principal:")
    print("  [1] Artigos PRISMA e revisão estruturada — briefing → PRISMA → PDF final")
    print("  [2] Documentos acadêmicos gerais — atividade, resumo, paper, dissertação")
    print("  [3] Projetos, TOML e insumos")
    print("  [4] Validação e prompts")
    print("  [5] Produção técnica")
    print("  [6] Saídas e relatórios")
    print("  [0] Sair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
