#!/usr/bin/env python3
# Alias canônico AP-004B para academic_pipeline_rc10.py.
# O orquestrador histórico permanece congelado byte a byte pela AP-003G.

from __future__ import annotations

import pathlib as _ap004b_alias_pathlib

_ap004b_alias_historical = _ap004b_alias_pathlib.Path(__file__).with_name(
    'academic_pipeline_rc10.py'
)
_ap004b_alias_source = _ap004b_alias_historical.read_bytes()
exec(
    compile(
        _ap004b_alias_source,
        str(_ap004b_alias_historical),
        "exec",
    ),
    globals(),
    globals(),
)

del _ap004b_alias_source
del _ap004b_alias_historical
del _ap004b_alias_pathlib
