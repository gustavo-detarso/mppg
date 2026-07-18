from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import latex_compile
from corpus_manager import (
    SourceDoc,
    _dedupe_messages,
    collect_orientation_docs,
    copy_documents_to_fulltext_cache,
    discover_local_documents,
    read_text_file_with_diagnostics,
    safe_extract_zip,
)


@pytest.mark.parametrize("engine", ["lualatex", "xelatex", "pdflatex"])
def test_safe_engine_accepts_supported_binary(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latex_compile.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert latex_compile._safe_engine(engine.upper()) == engine


def test_safe_engine_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latex_compile.shutil, "which", lambda name: f"/usr/bin/{name}")

    with pytest.raises(ValueError, match="pdf_engine inválido"):
        latex_compile._safe_engine("tectonic")


def test_safe_engine_requires_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latex_compile.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="não encontrado"):
        latex_compile._safe_engine("lualatex")


def test_compile_sequence_rejects_missing_org(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        latex_compile.run_compile_sequence(tmp_path / "ausente.org")


def test_compile_sequence_requires_emacs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text("* Texto\n", encoding="utf-8")
    monkeypatch.setattr(latex_compile.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="Emacs"):
        latex_compile.run_compile_sequence(org)


def test_compile_sequence_writes_elisp_and_returns_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text("* Texto\n", encoding="utf-8")
    extra = tmp_path / "latex"
    extra.mkdir()
    academic = tmp_path / "academic-writing.el"
    academic.write_text(";; config", encoding="utf-8")
    seen: dict[str, object] = {}

    def which(name: str):
        return f"/usr/bin/{name}"

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        seen["cwd"] = kwargs["cwd"]
        org.with_suffix(".pdf").write_bytes(b"%PDF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(latex_compile.shutil, "which", which)
    monkeypatch.setattr(latex_compile.subprocess, "run", run)

    result = latex_compile.run_compile_sequence(
        org,
        academic_writing=academic,
        latex_extra_path=extra,
        pdf_engine="xelatex",
    )

    elisp = tmp_path / "doc_export_pdf.el"
    text = elisp.read_text(encoding="utf-8")
    assert result == org.with_suffix(".pdf")
    assert 'load-file' in text
    assert "xelatex -interaction nonstopmode" in text
    assert '"biber %b"' in text
    assert str(extra.resolve()) + "//:" in seen["env"]["TEXINPUTS"]


def test_compile_sequence_writes_diagnostic_on_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org = tmp_path / "doc.org"
    org.write_text("* Texto\n", encoding="utf-8")
    monkeypatch.setattr(latex_compile.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        latex_compile.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=3, stdout="saida", stderr="erro"),
    )

    with pytest.raises(RuntimeError, match="Falha ao exportar PDF"):
        latex_compile.run_compile_sequence(org)

    diagnostic = tmp_path / "doc_pdf_erro.txt"
    assert "saida" in diagnostic.read_text(encoding="utf-8")
    assert "STDERR:\nerro" in diagnostic.read_text(encoding="utf-8")


def test_safe_extract_zip_extracts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "fontes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pasta/a.txt", "texto")
        zf.writestr("pasta/", "")

    extracted = safe_extract_zip(archive, tmp_path / "saida")

    assert [path.relative_to(tmp_path / "saida").as_posix() for path in extracted] == [
        "pasta/a.txt"
    ]
    assert extracted[0].read_text(encoding="utf-8") == "texto"


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "inseguro.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../fora.txt", "não")

    with pytest.raises(RuntimeError, match="Entrada insegura"):
        safe_extract_zip(archive, tmp_path / "saida")


def test_dedupe_messages_preserves_order_and_limit() -> None:
    assert _dedupe_messages([" A ", "", "B", "A", "C"], limit=2) == ["A", "B"]


def test_read_text_file_truncates_text_source(tmp_path: Path) -> None:
    source = tmp_path / "texto.md"
    source.write_text("abcdefghij", encoding="utf-8")

    text, warnings = read_text_file_with_diagnostics(source, max_chars=5)

    assert text == "abcde…"
    assert len(text) == 6
    assert warnings == []


def test_read_text_file_rejects_unknown_extension(tmp_path: Path) -> None:
    source = tmp_path / "dados.bin"
    source.write_bytes(b"x")

    with pytest.raises(RuntimeError, match="Extensão não suportada"):
        read_text_file_with_diagnostics(source)


def test_discover_local_documents_filters_suffixes_and_records_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "entrada"
    source.mkdir()
    good = source / "a.txt"
    bad = source / "b.md"
    ignored = source / "c.exe"
    good.write_text("A", encoding="utf-8")
    bad.write_text("B", encoding="utf-8")
    ignored.write_text("C", encoding="utf-8")

    def read(path: Path, _max_chars: int):
        if path.name == "b.md":
            raise RuntimeError("falha simulada")
        return path.read_text(encoding="utf-8"), []

    monkeypatch.setattr("corpus_manager.read_text_file_with_diagnostics", read)
    cfg = {
        "__config_dir__": str(tmp_path),
        "documentos_locais": {
            "input_dir": "entrada",
            "tipos": ["txt", "md"],
        },
    }

    docs, info = discover_local_documents(cfg, tmp_path / "work")

    assert [doc.label for doc in docs] == ["a.txt", "b.md"]
    assert docs[0].kind == "documento_base"
    assert docs[1].kind == "documento_base_erro"
    assert docs[1].metadata["error"] == "falha simulada"
    assert info["input_dir"] == str(source.resolve())


def test_discover_local_documents_requires_valid_input(tmp_path: Path) -> None:
    cfg = {
        "__config_dir__": str(tmp_path),
        "documentos_locais": {},
    }

    with pytest.raises(RuntimeError, match="input_zip ou input_dir"):
        discover_local_documents(cfg, tmp_path / "work")


def test_collect_orientation_docs_combines_files_and_inline(tmp_path: Path) -> None:
    orientation = tmp_path / "orientacao.txt"
    orientation.write_text("Use texto corrido.", encoding="utf-8")
    cfg = {
        "__config_dir__": str(tmp_path),
        "orientacoes": {
            "paths": ["orientacao.txt"],
            "inline": "Evite listas.",
        },
    }

    docs = collect_orientation_docs(cfg, tmp_path / "work")

    assert [(doc.kind, doc.label) for doc in docs] == [
        ("orientacao", "orientacao.txt"),
        ("orientacao_inline", "Orientação inline"),
    ]
    assert docs[1].extracted_text == "Evite listas."


def test_copy_documents_to_cache_renames_collisions_and_updates_metadata(
    tmp_path: Path,
) -> None:
    one = tmp_path / "a" / "fonte.txt"
    two = tmp_path / "b" / "fonte.txt"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_text("um", encoding="utf-8")
    two.write_text("dois", encoding="utf-8")
    docs = [
        SourceDoc(str(one), "documento_base", "fonte.txt", "um"),
        SourceDoc(str(two), "documento_base", "fonte.txt", "dois"),
    ]

    copied = copy_documents_to_fulltext_cache(docs, tmp_path / "cache")

    assert [path.name for path in copied] == ["fonte.txt", "fonte_2.txt"]
    assert all("fulltext_cache_path" in doc.metadata for doc in docs)


def test_copy_documents_to_cache_cleans_previous_contents(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "antigo.txt").write_text("antigo", encoding="utf-8")

    copied = copy_documents_to_fulltext_cache([], cache, clean=True)

    assert copied == []
    assert list(cache.iterdir()) == []
