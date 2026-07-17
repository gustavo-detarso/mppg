from __future__ import annotations

import ast
import importlib
import inspect
import json
import pathlib
import textwrap
from collections import defaultdict
from typing import Any

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]

PLAN = (
    REPOSITORY_ROOT
    / "docs/refactor/academic-pipeline/AP-005/"
    "ap005b2_prisma_adapter_batches.json"
)

RC10 = (
    ROOT
    / "app_bundle/scripts/pipeline/"
    "academic_pipeline_rc10.py"
)

MODULE_NAME = (
    "academic_pipeline.prisma_generic_orchestration"
)


def _load_plan() -> dict[str, Any]:
    return json.loads(
        PLAN.read_text(encoding="utf-8")
    )


def _module() -> Any:
    return importlib.import_module(MODULE_NAME)


def _single_return_call(function: Any) -> ast.Call:
    source = textwrap.dedent(
        inspect.getsource(function)
    )
    tree = ast.parse(source)

    function_node = tree.body[0]

    assert isinstance(
        function_node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
    )

    body = list(function_node.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    assert len(body) == 1
    assert isinstance(body[0], ast.Return)
    assert isinstance(body[0].value, ast.Call)

    return body[0].value


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _rc10_imports() -> dict[
    tuple[str, str],
    list[str],
]:
    tree = ast.parse(
        RC10.read_text(encoding="utf-8"),
        filename=str(RC10),
    )

    result: dict[
        tuple[str, str],
        list[str],
    ] = defaultdict(list)

    for function in tree.body:
        if not isinstance(
            function,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        for node in ast.walk(function):
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.module != MODULE_NAME:
                continue

            for alias in node.names:
                local_name = alias.asname or alias.name

                result[
                    (function.name, local_name)
                ].append(alias.name)

    return result


def _adapter_presence(
    module: Any,
    entries: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        entry["candidate_name"]: hasattr(
            module,
            entry["candidate_name"],
        )
        for entry in entries
    }


def _arguments_for(
    signature: inspect.Signature,
) -> tuple[list[Any], dict[str, Any]]:
    runtime = {
        "runtime-marker": object(),
    }

    positional: list[Any] = []
    keywords: dict[str, Any] = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        kind = parameter.kind

        if kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if name == "runtime":
                positional.append(runtime)
            else:
                positional.append(object())

        elif kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(
                [object(), object()]
            )

        elif kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[name] = object()

        elif kind is inspect.Parameter.VAR_KEYWORD:
            keywords["extra_one"] = object()
            keywords["extra_two"] = object()

        else:
            raise AssertionError(kind)

    return positional, keywords


def test_ap005b2_rollout_state_is_batch_atomic() -> None:
    payload = _load_plan()
    module = _module()
    entries = payload["entries"]
    presence = _adapter_presence(module, entries)

    for batch in payload["batches"]:
        batch_entries = [
            entry
            for entry in entries
            if entry["batch"] == batch["batch"]
        ]

        present_count = sum(
            presence[entry["candidate_name"]]
            for entry in batch_entries
        )

        assert present_count in {
            0,
            len(batch_entries),
        }, (
            f"{batch['batch']} parcialmente aplicado: "
            f"{present_count}/{len(batch_entries)}"
        )


def test_ap005b2_adapters_and_wrappers_follow_contract() -> None:
    payload = _load_plan()
    module = _module()

    exported = set(module.__all__)
    protected = set(
        module._PROTECTED_RUNTIME_NAMES
    )

    for entry in payload["entries"]:
        wrapper = getattr(
            module,
            entry["wrapper_name"],
        )

        assert callable(wrapper)
        assert entry["wrapper_name"] in exported
        assert entry["wrapper_name"] in protected
        assert entry["body_function"] in protected
        assert hasattr(
            module,
            entry["body_function"],
        )

        candidate = getattr(
            module,
            entry["candidate_name"],
            None,
        )

        wrapper_call = _single_return_call(wrapper)

        if candidate is None:
            assert (
                _call_name(wrapper_call)
                == "_invoke_with_runtime"
            )
            continue

        assert callable(candidate)
        assert inspect.signature(candidate) == (
            inspect.signature(wrapper)
        )
        assert entry["candidate_name"] in exported
        assert entry["candidate_name"] in protected

        candidate_call = _single_return_call(
            candidate
        )

        assert (
            _call_name(candidate_call)
            == "_invoke_with_runtime"
        )
        assert (
            _call_name(wrapper_call)
            == entry["candidate_name"]
        )

        assert (
            ast.unparse(candidate_call.args[0])
            == entry["body_function"]
        )
        assert (
            ast.unparse(candidate_call.args[1])
            == "runtime"
        )


def test_ap005b2_rc10_consumers_follow_rollout_state() -> None:
    payload = _load_plan()
    module = _module()
    imports = _rc10_imports()

    for entry in payload["entries"]:
        key = (
            entry["rc10_consumer_function"],
            entry["rc10_local_alias"],
        )

        imported_names = imports.get(key, [])

        assert len(imported_names) == 1, (
            f"{key}: imports={imported_names}"
        )

        candidate_exists = hasattr(
            module,
            entry["candidate_name"],
        )

        expected_name = (
            entry["candidate_name"]
            if candidate_exists
            else entry["wrapper_name"]
        )

        assert imported_names == [expected_name]


def test_ap005b2_runtime_equivalence_for_migrated_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _load_plan()
    module = _module()

    migrated = [
        entry
        for entry in payload["entries"]
        if hasattr(
            module,
            entry["candidate_name"],
        )
    ]

    captures: list[
        tuple[Any, Any, tuple[Any, ...], dict[str, Any]]
    ] = []

    marker = object()

    def fake_invoke(
        function: Any,
        runtime: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object:
        captures.append(
            (function, runtime, args, kwargs)
        )
        return marker

    monkeypatch.setattr(
        module,
        "_invoke_with_runtime",
        fake_invoke,
    )

    for entry in migrated:
        candidate = getattr(
            module,
            entry["candidate_name"],
        )
        wrapper = getattr(
            module,
            entry["wrapper_name"],
        )

        positional, keywords = _arguments_for(
            inspect.signature(candidate)
        )

        captures.clear()

        candidate_result = candidate(
            *positional,
            **keywords,
        )

        assert candidate_result is marker
        assert len(captures) == 1

        candidate_capture = captures[0]

        captures.clear()

        wrapper_result = wrapper(
            *positional,
            **keywords,
        )

        assert wrapper_result is marker
        assert len(captures) == 1
        assert captures[0] == candidate_capture
