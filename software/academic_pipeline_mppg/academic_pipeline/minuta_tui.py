"""MPPG_MINUTA_TUI_V251: Minuta somente no submenu de Documentos acadêmicos gerais.

A integração detecta a categoria [2] pela estrutura real das opções do menu,
NUNCA pelo texto de status reutilizado em outras telas. Não modifica a TUI
original e restaura sempre o objeto ui.menu em finally.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

_KEY = "mppg_minuta_fgv_v25"
_LABEL = "Minuta FGV — criar, editar, validar e compilar"
_STATE = Path.home() / ".config" / "academic_pipeline_mppg" / "minuta_fgv_tui.json"
_DRAFT_MARK = "MPPG_MINUTA_TUI_RASCUNHO"


def _norm_label(value: Any) -> str:
    """Normalização restrita a rótulos de opções (não analisa textos de status)."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)).casefold()
                   if not unicodedata.combining(c))


def _document_category_key(options: Any) -> str | None:
    """Retorna a chave da categoria [2] SOMENTE no menu raiz conhecido.

    Os menus secundários podem receber o mesmo status_text da tela principal;
    por isso títulos e textos gerais não servem para identificar o menu.
    """
    if not isinstance(options, (tuple, list)):
        return None
    if not all(isinstance(it, (tuple, list)) and len(it) == 3 and
               isinstance(it[0], str) for it in options):
        return None
    labels = [(_norm_label(it[1]), it[0]) for it in options]
    docs = [key for label, key in labels if "documentos academicos gerais" in label]
    prisma = any("artigos prisma" in label for label, _ in labels)
    project = any("projetos" in label and "toml" in label for label, _ in labels)
    if len(docs) == 1 and prisma and project:
        return docs[0]
    return None


def _general_documents_submenu(ui: Any, original: Callable[..., str | None],
                               docs_key: str) -> bool:
    """False: voltar à raiz; True: continuar para o fluxo [2] já existente."""
    choices = [
        ("original", "[1] Atividade, resumo, paper ou dissertação — fluxo existente", ["1"]),
        ("minuta", "[2] Minuta FGV — criar, editar, validar e compilar", ["2"]),
        ("back", "[0] Voltar ao menu principal", ["0"]),
    ]
    while True:
        choice = original(
            "DOCUMENTOS ACADÊMICOS GERAIS — FGV",
            "Escolha um tipo de documento. Os outros fluxos acadêmicos permanecem intactos.",
            choices,
        )
        if choice in (None, "back"):
            return False
        if choice == "original":
            return True
        if choice == "minuta":
            _minuta_menu(ui, original)
        else:
            # Retorno inesperado não ativa outra categoria.
            ui.message("Minuta FGV", f"Opção desconhecida no submenu: {choice!r}")


def _load_state() -> Path | None:
    try:
        data = json.loads(_STATE.read_text(encoding="utf-8"))
        path = Path(str(data["config"])).expanduser().resolve()
        return path if path.is_file() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_state(config: Path) -> None:
    _STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Não grava o conteúdo da dissertação nem metadados sensíveis; somente o caminho.
    tmp = _STATE.with_name(_STATE.name + ".tmp")
    tmp.write_text(json.dumps({"config": str(config.resolve())}, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, _STATE)


def _ask(ui: Any, title: str, label: str, default: str = "", **kw: Any) -> str | None:
    result = ui.input_text(title, label, default=default, **kw)
    if result is None:
        return None
    return str(result).strip()


def _message(ui: Any, text: str, *, title: str = "Minuta FGV") -> None:
    ui.message(title, text[:12000])


def _slug(name: str) -> str:
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    slug = re.sub("[^A-Za-z0-9_-]+", "_", plain).strip("_-" ).lower()[:80]
    return slug or "minuta_nova"


def _run_minuta(args: list[str]) -> tuple[int, str]:
    """Reutiliza o mesmo motor da CLI, sem shell, sem regeneração por IA."""
    from . import minuta_runtime
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = minuta_runtime.main(args)
        return int(code), out.getvalue()
    except Exception as exc:
        return 2, out.getvalue() + f"\nERRO: {type(exc).__name__}: {exc}"


def _generic_draft() -> str:
    return (f"# {_DRAFT_MARK}: substitua os comentários antes de compilar.\n"
            "* Tema\n\n# Delimite o tema e a unidade de análise.\n\n"
            "* Problema\n\n# Apresente a pergunta de pesquisa e seu contexto.\n\n"
            "* Protocolo da revisão\n\n# Especifique delineamento, bases, critérios e síntese.\n")


def _new_minuta(ui: Any) -> Path | None:
    base = Path.home() / "Documentos" / "mppg" / "dissertacao" / "minuta_nova"
    path_raw = _ask(ui, "Nova minuta FGV", "Pasta do novo projeto:", str(base), path_completion=True, only_directories=False)
    if not path_raw:
        return None
    project = Path(path_raw).expanduser().resolve()
    if (project / "minuta.toml").exists():
        _message(ui, "Já existe minuta.toml nessa pasta. Use 'Selecionar minuta existente' para preservá-lo.")
        return None
    name = _ask(ui, "Nova minuta FGV", "Identificador (sem espaços ou acentos):", _slug(project.name))
    if not name:
        return None
    name = _slug(name)
    title = _ask(ui, "Nova minuta FGV", "Título do documento:", "Minuta inicial de dissertação")
    if not title:
        return None
    author = _ask(ui, "Nova minuta FGV", "Nome do mestrando:")
    if not author:
        _message(ui, "O nome do mestrando é obrigatório.")
        return None
    adviser = _ask(ui, "Nova minuta FGV", "Professor orientador:")
    if adviser is None:
        return None
    date_str = _ask(ui, "Nova minuta FGV", "Data que aparecerá na identificação:", dt.date.today().strftime("%d/%m/%Y"))
    if date_str is None:
        return None
    # Não substituir arquivos do usuário, inclusive caso o Org exista isoladamente.
    existed_org = (project / "conteudo.org").exists()
    if existed_org:
        _message(ui, "Já existe conteudo.org. Para evitar misturar projetos, escolha uma pasta vazia.")
        return None
    rc, result = _run_minuta(["init", "--project", str(project), "--name", name,
                              "--title", title, "--author", author,
                              "--adviser", adviser, "--date", date_str])
    if rc:
        _message(ui, result, title="Falha ao criar minuta")
        return None
    org = project / "conteudo.org"
    # O init v2.4 copia o exemplo de PMF; somente no NOVO projeto TUI trocamos
    # este exemplo (ainda não editado) por um roteiro genérico.
    sample = Path(__file__).resolve().parent.parent / "app_bundle/institutions/fgv/templates/exemplo_conteudo_minuta.org"
    if not sample.is_file() or org.read_bytes() != sample.read_bytes():
        _message(ui, "Minuta criada. O Org inicial não corresponde ao exemplo do pacote; não foi substituído.\n" + result)
    else:
        org.write_text(_generic_draft(), encoding="utf-8")
        _message(ui, "Projeto criado. O conteúdo começa com um roteiro genérico; edite o Org antes de compilar.\n\n" + result)
    config = project / "minuta.toml"
    _save_state(config)
    return config


def _choose_minuta(ui: Any, active: Path | None) -> Path | None:
    default = str(active or (Path.home() / "Documentos/mppg/dissertacao/minuta_buta_nativa/minuta.toml"))
    raw = _ask(ui, "Abrir minuta", "Caminho do arquivo minuta.toml:", default,
               path_completion=True, allowed_suffixes=(".toml",))
    if raw is None:
        return active
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        _message(ui, f"Arquivo não encontrado: {path}")
        return active
    rc, result = _run_minuta(["check", "--config", str(path)])
    if rc:
        _message(ui, result, title="Minuta não validada")
        return active
    _save_state(path)
    _message(ui, f"Minuta selecionada:\n{path}\n\n{result}")
    return path


def _edit(ui: Any, path: Path) -> None:
    editor = os.environ.get("EDITOR", "").strip()
    if editor:
        cmd = shlex.split(editor)
    elif shutil.which("emacs"):
        cmd = ["emacs", "-nw"]
    elif shutil.which("nano"):
        cmd = ["nano"]
    else:
        _message(ui, "Nenhum editor encontrado. Configure EDITOR=\"emacs -nw\" ou instale um editor de terminal.")
        return
    if not cmd or not shutil.which(cmd[0]):
        _message(ui, f"Editor indisponível: {cmd[0] if cmd else editor}")
        return
    try:
        rc = subprocess.run([*cmd, str(path)], cwd=path.parent, check=False).returncode
        if rc:
            _message(ui, f"O editor retornou código {rc}. Confira o arquivo: {path}")
    except OSError as exc:
        _message(ui, f"Não foi possível abrir o editor: {exc}")


def _read_source(config: Path) -> Path:
    from .minuta_runtime import _load
    _, _, source, _ = _load(config)
    return source


def _build(ui: Any, config: Path) -> None:
    from .minuta_runtime import _load
    try:
        cfg, _, source, output = _load(config)
    except (OSError, ValueError) as exc:
        _message(ui, str(exc), title="Configuração inválida")
        return
    if _DRAFT_MARK in source.read_text(encoding="utf-8"):
        _message(ui, "O projeto ainda contém o roteiro genérico. Edite o conteúdo Org antes de compilar.")
        return
    rc, result = _run_minuta(["check", "--config", str(config)])
    if rc:
        _message(ui, result, title="Falha na validação")
        return
    dest = output / cfg["_internal"]["nome"]
    args = ["build", "--config", str(config)]
    if dest.exists():
        if not ui.confirm("Recompilar minuta", "Já existe uma compilação. Guardar backup e substituir a saída final?", default=False):
            return
        args.append("--overwrite")
    if not ui.confirm("Compilar minuta FGV", "Gerar PDF e DOCX pelo motor do MPPG?", default=True):
        return
    rc, result = _run_minuta(args)
    _message(ui, result, title="Compilação concluída" if rc == 0 else "Compilação não aprovada")


def _outputs(ui: Any, config: Path, old_menu: Callable[..., str | None]) -> None:
    from .minuta_runtime import _load
    try:
        cfg, _, _, output = _load(config)
    except (OSError, ValueError) as exc:
        _message(ui, str(exc))
        return
    dest = output / cfg["_internal"]["nome"]
    pdf, docx = dest / f"{cfg['_internal']['nome']}.pdf", dest / f"{cfg['_internal']['nome']}.docx"
    options = [("pdf", "Abrir PDF", ["p"]), ("docx", "Abrir DOCX", ["d"]),
               ("folder", "Abrir pasta de saída", ["s"]), ("back", "Voltar", ["v"])]
    choice = old_menu("Minuta FGV — saídas", f"Pasta: {dest}\nPDF: {'OK' if pdf.exists() else 'ausente'}\nDOCX: {'OK' if docx.exists() else 'ausente'}", options)
    target = {"pdf": pdf, "docx": docx, "folder": dest}.get(choice)
    if not target:
        return
    if not target.exists():
        _message(ui, f"Arquivo ainda não foi gerado: {target}")
        return
    opener = shutil.which("xdg-open")
    if opener is None:
        _message(ui, f"Não encontrei xdg-open. Caminho do arquivo:\n{target}")
        return
    try:
        subprocess.Popen([opener, str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=dest)
    except OSError as exc:
        _message(ui, f"Falha ao abrir {target}: {exc}")


def _minuta_menu(ui: Any, old_menu: Callable[..., str | None]) -> None:
    active = _load_state()
    while True:
        options = [
            ("new", "Nova minuta (criar TOML e Org)", ["n"]),
            ("choose", "Selecionar minuta existente", ["s"]),
            ("toml", "Editar minuta.toml", ["t"]),
            ("org", "Editar conteudo.org", ["e"]),
            ("check", "Validar minuta", ["v"]),
            ("build", "Compilar PDF e DOCX", ["c"]),
            ("outputs", "Consultar/abrir saídas", ["a"]),
            ("back", "Voltar à Central Operacional", ["q"]),
        ]
        choice = old_menu("MINUTA FGV — CENTRAL OPERACIONAL",
                          f"Modelo institucional v2.4 | Fluxo TUI v2.5\nProjeto ativo: {active or '(nenhum)'}\n"
                          "Criar → editar Org/TOML → validar → compilar → conferir.", options)
        if choice in (None, "back"):
            return
        if choice == "new":
            active = _new_minuta(ui) or active
        elif choice == "choose":
            active = _choose_minuta(ui, active)
        elif not active:
            _message(ui, "Selecione uma minuta existente ou crie uma nova antes de prosseguir.")
        elif choice == "toml":
            _edit(ui, active)
        elif choice == "org":
            try:
                _edit(ui, _read_source(active))
            except (OSError, ValueError) as exc:
                _message(ui, str(exc))
        elif choice == "check":
            rc, result = _run_minuta(["check", "--config", str(active)])
            _message(ui, result, title="Validação OK" if rc == 0 else "Validação falhou")
        elif choice == "build":
            _build(ui, active)
        elif choice == "outputs":
            _outputs(ui, active, old_menu)


def run_with_minuta_menu(run_existing_tui: Callable[[], int]) -> int:
    """Acrescenta Minuta apenas como subitem de [2] Documentos acadêmicos gerais."""
    from app_bundle.scripts.pipeline import academic_pipeline_tui as tui_mod
    ui = getattr(tui_mod, "ui", None)
    if ui is None or not callable(getattr(ui, "menu", None)):
        raise RuntimeError("Interface da TUI mudou: não existe ui.menu")
    original = ui.menu
    root_seen = False

    def extended(*args: Any, **kwargs: Any) -> str | None:
        nonlocal root_seen
        options = args[2] if len(args) >= 3 else kwargs.get("options")
        key = _document_category_key(options)
        if key is None:
            # Preserva integralmente TODOS os menus secundários.
            return original(*args, **kwargs)
        root_seen = True
        while True:
            selected = original(*args, **kwargs)
            if selected != key:
                return selected
            # Só após selecionar a categoria [2] aparece a opção Minuta.
            # O fluxo original continua por sua própria chave, sem reimplementá-lo.
            if _general_documents_submenu(ui, original, key):
                return key
            # Se o usuário pediu "Voltar", reabre o menu raiz.

    try:
        ui.menu = extended
        rc = run_existing_tui()
        if not root_seen:
            print("AVISO MPPG MINUTA v2.5.1: lista de categorias do menu principal não reconhecida; nenhum menu alterado.", file=sys.stderr)
        return int(rc)
    finally:
        ui.menu = original
