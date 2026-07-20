from __future__ import annotations
import importlib.util, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
MANIFEST = REPO / "docs/refactor/academic-pipeline/AP-006/ap006d4c_cache_regeneration_contract.json"
VALIDATOR = REPO / "tools/refactor/ap006d4c_validate_cache_regeneration.py"
def load_validator():
    spec = importlib.util.spec_from_file_location("ap006d4c_validator", VALIDATOR); assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def test_manifest_locks_the_cache_regeneration_decision() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["phase"] == "AP-006D.4C" and data["summary"]["pair_count"] == 4
    assert data["summary"]["byte_equal_pair_count"] == 4 and data["summary"]["total_data_row_count"] == 308
    assert data["summary"]["total_old_reference_line_count"] == 308
    assert data["validated_dry_run"]["regenerated_count"] == 4 and data["validated_dry_run"]["clone_clean_after_regeneration"] is True
    assert data["constraints"]["authoritative_source_review_deferred_to_ap006d4d"] is True
    assert data["constraints"]["manual_cache_edit_forbidden"] is True
    assert data["productive_contract"]["targeted_regeneration_policy"]["clean"] is False
def test_validator_confirms_pairs_and_productive_primitive() -> None:
    result = load_validator().validate(REPO, MANIFEST)
    assert result["status"] == "ok" and result["pair_count"] == 4 and result["exact_match_count"] == 4
    assert result["copy_primitive"] == "shutil.copy2"
