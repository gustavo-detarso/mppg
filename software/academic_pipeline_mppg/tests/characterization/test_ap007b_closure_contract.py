from __future__ import annotations
import ast,importlib.util,json
from pathlib import Path
def root(): return Path(__file__).resolve().parents[4]
def validator():
 p=root()/"tools/refactor/ap007b_validate_closure.py"; s=importlib.util.spec_from_file_location("ap007b_closure",p); assert s and s.loader; m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def test_ap007b3_validator_accepts_state():
 r=validator().validate(root(),"auto"); assert r["status"]=="ok" and r["gate_ap007b_commit"]=="PASS" and r["candidate_path_count"]==10 and r["closure_artifact_count"]==4
def test_ap007b3_manifest_is_exact():
 p=json.loads((root()/"docs/refactor/academic-pipeline/AP-007/ap007b_closure_manifest.json").read_text(encoding="utf-8")); assert p["phase"]=="AP-007B.3"; assert p["validation"]["contract"]=={"passed":20}; assert p["validation"]["source_tree_regressions"]=={"passed":24,"deselected":4}; assert p["commit_readiness"]["candidate_path_count"]==10; assert p["commit_readiness"]["explicit_authorization_required"] is True
def test_ap007b3_topology_is_native_first_wave():
 c=(root()/"software/academic_pipeline_mppg/academic_pipeline/cli.py").read_text(encoding="utf-8"); r=(root()/"software/academic_pipeline_mppg/academic_pipeline/runtime.py").read_text(encoding="utf-8"); assert "from .runtime import run" in c and "from .legacy import run_legacy" in c and "return run(argv, legacy_runner=run_legacy)" in c and "FIRST_WAVE_OPTIONS" in r; t=ast.parse(r); assert not [n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {"globals","locals","eval","exec","__import__"}]
def test_ap007b3_distribution_boundary():
 p=json.loads((root()/"docs/refactor/academic-pipeline/AP-007/ap007b_closure_manifest.json").read_text(encoding="utf-8")); assert p["deferred_to_ap007e"]=={"isolated_direct_script_tests":4,"reason":"distribution_and_installation_contract"}; assert p["commit_readiness"]["tag_authorized"] is False
