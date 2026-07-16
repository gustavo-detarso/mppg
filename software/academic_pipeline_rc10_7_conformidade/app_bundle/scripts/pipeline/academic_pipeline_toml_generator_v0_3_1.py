#!/usr/bin/env python3
# Wrapper transitório AP-004B para academic_pipeline_toml_generator.py.
# A implementação canônica é executada no namespace histórico.

from __future__ import annotations

import pathlib as _ap004b_compat_pathlib

_ap004b_compat_canonical = _ap004b_compat_pathlib.Path(__file__).with_name(
    'academic_pipeline_toml_generator.py'
)
_ap004b_compat_source = _ap004b_compat_canonical.read_bytes()
exec(
    compile(
        _ap004b_compat_source,
        str(_ap004b_compat_canonical),
        "exec",
    ),
    globals(),
    globals(),
)

del _ap004b_compat_source
del _ap004b_compat_canonical
del _ap004b_compat_pathlib
