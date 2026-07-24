#!/usr/bin/env python3
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'software/academic_pipeline_mppg/academic_pipeline/list_profiles_runtime.py'
M=ROOT/'docs/refactor/academic-pipeline/AP-007/ap007d2_list_profiles_native_adapter.json'
d=json.loads(M.read_text(encoding='utf-8')); s=A.read_text(encoding='utf-8'); ast.parse(s)
assert d['status']=='materialized_route_still_legacy'
assert hashlib.sha256(s.encode()).hexdigest()==d['adapter']['sha256']
assert 'globals(' not in s and 'locals(' not in s and 'importlib' not in s
assert d['adapter']['dependency_strategy']=='ast_transitive_top_level_closure_with_safe_classes'
assert d['source']['path']=='software/academic_pipeline_mppg/app_bundle/scripts/pipeline/academic_pipeline_toml_generator_interativo.py'
assert 'backup' not in d['source']['path'].lower()
assert [item['name'] for item in d['adapter']['classes']]==['Preset']
print('[OK] AP-007D.2 validator')
print('command=--list-profiles')
print('status=materialized_route_still_legacy')
