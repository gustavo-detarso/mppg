from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import document_translation as translation
import mindmap_manager
from document_model import AcademicDocument, Block, DocumentMetadata, Section


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("English", ("en", "English")),
        ("inglês", ("en", "English")),
        ("español", ("es", "Español")),
        ("pt-BR", ("pt-br", "pt-BR")),
        ("Português do Brasil", ("portugu-s-do-brasil", "Português do Brasil")),
    ],
)
def test_normalize_language_preserves_current_alias_contract(
    raw: str,
    expected: tuple[str, str],
) -> None:
    assert translation.normalize_language(raw) == expected


def test_normalize_language_rejects_empty_value() -> None:
    with pytest.raises(translation.TranslationError, match="vazio"):
        translation.normalize_language("")


def test_requested_translation_languages_excludes_principal_and_duplicates() -> None:
    cfg = {
        "idiomas_saida": {
            "gerar_traducao_ia": True,
            "principal": "pt-BR",
            "idiomas_adicionais": ["en", "pt-BR", "English", "es"],
        }
    }

    assert translation.requested_translation_languages(cfg) == [
        ("en", "English"),
        ("es", "Español"),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(100, 2500), (5000, 5000), (99999, 24000), ("inválido", 12000)],
)
def test_translation_batch_size_is_clamped(raw: object, expected: int) -> None:
    cfg = {"idiomas_saida": {"max_chars_por_lote": raw}}
    assert translation.translation_batch_size(cfg) == expected


def test_collect_translatable_strings_skips_protected_branches_and_names() -> None:
    payload = {
        "metadata": {
            "titulo": "Título acadêmico",
            "autor": "Nome Próprio",
            "doi": "10.1000/x",
        },
        "sections": [
            {
                "title": "Introdução",
                "blocks": [{"text": "Texto substantivo para tradução."}],
            }
        ],
        "bibliography": {"entries_used": ["silva2020"]},
        "diagnostics": {"message": "Não traduzir diagnóstico."},
    }

    items = translation.collect_translatable_strings(payload)
    paths = [path for path, _text in items]

    assert ("metadata", "titulo") in paths
    assert ("sections", 0, "title") in paths
    assert ("sections", 0, "blocks", 0, "text") in paths
    assert ("metadata", "autor") not in paths
    assert not any(path[0] in {"bibliography", "diagnostics"} for path in paths)


def test_chunk_items_assigns_stable_ids_and_respects_limit() -> None:
    items = [(("a",), "x" * 50), (("b",), "y" * 50)]

    chunks = translation._chunk_items(items, max_chars=200)

    assert len(chunks) == 2
    assert chunks[0][0][0] == "t00001"
    assert chunks[1][0][0] == "t00002"


def test_translation_batch_falls_back_when_response_format_is_rejected() -> None:
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise RuntimeError("unsupported")
            content = json.dumps(
                {"translations": [{"id": "t00001", "text": "Translated"}]}
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = translation._request_translation_batch(
        client,
        "model",
        "English",
        [("t00001", ("text",), "Texto")],
    )

    assert result == {"t00001": "Translated"}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_translation_batch_rejects_missing_ids() -> None:
    content = json.dumps({"translations": []})
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_k: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )
    )

    with pytest.raises(translation.TranslationError, match="todos os campos"):
        translation._request_translation_batch(
            client,
            "model",
            "English",
            [("t00001", ("text",), "Texto")],
        )


class FakeDocument:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def model_dump(self, mode: str = "python") -> dict:
        return self.payload

    @classmethod
    def model_validate(cls, payload: dict):
        return cls(payload)


def test_translate_document_model_preserves_protected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(
        {
            "metadata": {"titulo": "Título", "autor": "Gustavo"},
            "sections": [{"title": "Introdução", "text": "Texto acadêmico."}],
            "bibliography": {"entries_used": ["silva2020"]},
        }
    )

    def fake_batch(_client, _model, _label, items):
        return {item_id: "EN:" + text for item_id, _path, text in items}

    monkeypatch.setattr(translation, "_request_translation_batch", fake_batch)

    translated, audit = translation.translate_document_model(
        object(),
        "model",
        document,
        "en",
        max_chars=5000,
    )

    assert translated.payload["metadata"]["titulo"] == "EN:Título"
    assert translated.payload["metadata"]["autor"] == "Gustavo"
    assert translated.payload["bibliography"]["entries_used"] == ["silva2020"]
    assert audit["idioma_codigo"] == "en"
    assert audit["referencias_preservadas"] is True
    assert audit["campos_traduzidos"] == 3


def _document() -> AcademicDocument:
    return AcademicDocument(
        metadata=DocumentMetadata(titulo="Documento"),
        sections=[
            Section(
                title="Introdução",
                blocks=[Block(type="paragraph", text="Texto acadêmico.")],
            )
        ],
    )


def test_normalize_level_colors_accepts_list_dict_and_default() -> None:
    assert mindmap_manager._normalize_level_colors(["#111111", ""]) == ["#111111"]
    assert mindmap_manager._normalize_level_colors(
        {"nivel_2": "#222222", "nivel_0": "#000000"}
    ) == ["#000000", "#222222"]
    assert mindmap_manager._normalize_level_colors(None) == (
        mindmap_manager.DEFAULT_MINDMAP_LEVEL_COLORS
    )


def test_colorize_mindmap_assigns_colors_by_level() -> None:
    source = "@startmindmap\n* Raiz\n** Ramo\n*** Folha\n@endmindmap\n"
    cfg = {
        "mapa_mental": {
            "cores_niveis": ["#111111", "#222222", "#333333"],
        }
    }

    colored = mindmap_manager.colorize_mindmap_plantuml(source, cfg)

    assert "*[#111111] Raiz" in colored
    assert "**[#222222] Ramo" in colored
    assert "***[#333333] Folha" in colored


def test_colorize_mindmap_can_preserve_existing_colors() -> None:
    source = "@startmindmap\n*[#ABCDEF] Raiz\n@endmindmap\n"
    cfg = {"mapa_mental": {"sobrescrever_cores_existentes": False}}

    assert mindmap_manager.colorize_mindmap_plantuml(source, cfg) == source


def test_sanitize_plantuml_extracts_fenced_mindmap() -> None:
    raw = "```plantuml\ntexto\n@startmindmap\n* Raiz\n@endmindmap\n```"

    assert mindmap_manager.sanitize_plantuml(raw) == (
        "@startmindmap\n* Raiz\n@endmindmap\n"
    )


def test_sanitize_plantuml_wraps_unframed_content() -> None:
    assert mindmap_manager.sanitize_plantuml("* Raiz") == (
        "@startmindmap\n* Raiz\n@endmindmap\n"
    )


def test_mindmap_artifact_paths_normalize_name_and_format(tmp_path: Path) -> None:
    cfg = {
        "mapa_mental": {
            "diretorio_imagens": "figuras",
            "arquivo": "Mapa de Evidências",
            "formato": "pdf",
        }
    }

    paths = mindmap_manager.mindmap_artifact_paths(cfg, tmp_path)

    assert paths["images_dir"] == tmp_path / "figuras"
    assert paths["puml"].name == "mapa_de_evidencias.puml"
    assert paths["image"].name == "mapa_de_evidencias.png"


def test_attach_mindmap_figure_replaces_previous_entry(tmp_path: Path) -> None:
    document = _document()
    image = tmp_path / "images" / "mapa.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    document.figures = [
        mindmap_manager.FigureSpec(
            id="mapa_mental",
            title="Antigo",
            path="old.png",
        )
    ]

    mindmap_manager.attach_mindmap_figure(document, {}, tmp_path, image)

    assert len([f for f in document.figures if f.id == "mapa_mental"]) == 1
    figure = document.figures[0]
    assert figure.path == "images/mapa.png"
    assert figure.placement == "after_references"
    assert figure.page_break_before is True


def test_render_or_generate_requires_client_without_existing_puml(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="Mapa mental ainda não possui"):
        mindmap_manager.render_or_generate_mindmap(
            None,
            "model",
            {"mapa_mental": {"gerar": True}},
            _document(),
            tmp_path,
        )


def test_render_plantuml_reports_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    puml = tmp_path / "mapa.puml"
    puml.write_text("@startmindmap\n* X\n@endmindmap\n", encoding="utf-8")
    monkeypatch.setattr(mindmap_manager.shutil, "which", lambda _name: None)
    monkeypatch.delenv("PLANTUML_JAR", raising=False)

    image, error = mindmap_manager.render_plantuml(puml, {})

    assert image is None
    assert "PlantUML não encontrado" in error


def test_render_plantuml_uses_command_and_detects_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    puml = tmp_path / "mapa.puml"
    puml.write_text("@startmindmap\n* X\n@endmindmap\n", encoding="utf-8")

    def run(cmd, **_kwargs):
        puml.with_suffix(".svg").write_text("<svg/>", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        mindmap_manager.shutil,
        "which",
        lambda name: "/usr/bin/plantuml" if name == "plantuml" else None,
    )
    monkeypatch.setattr(mindmap_manager.subprocess, "run", run)

    image, error = mindmap_manager.render_plantuml(
        puml,
        {"mapa_mental": {"formato": "svg"}},
    )

    assert image == puml.with_suffix(".svg")
    assert error is None


def test_delete_existing_mindmap_outputs_removes_known_formats(
    tmp_path: Path,
) -> None:
    cfg = {"mapa_mental": {"arquivo": "mapa", "formato": "png"}}
    images = tmp_path / "images"
    images.mkdir()
    for suffix in (".puml", ".png", ".svg"):
        (images / f"mapa{suffix}").write_text("x", encoding="utf-8")

    removed = mindmap_manager.delete_existing_mindmap_outputs(cfg, tmp_path)

    assert len(removed) == 3
    assert list(images.iterdir()) == []


def test_attach_existing_mindmap_reuses_image_without_ai(tmp_path: Path) -> None:
    cfg = {"mapa_mental": {"gerar": True, "arquivo": "mapa"}}
    image = tmp_path / "images" / "mapa.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    document = _document()

    result = mindmap_manager.attach_existing_mindmap_if_available(
        document,
        cfg,
        tmp_path,
    )

    assert result["reused"] is True
    assert result["image_path"] == str(image)
    assert document.figures[0].path == "images/mapa.png"


def test_generate_mindmap_short_circuits_when_disabled(tmp_path: Path) -> None:
    assert mindmap_manager.generate_and_attach_mindmap(
        object(),
        "model",
        {},
        _document(),
        tmp_path,
    ) is None
