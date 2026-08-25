#!/usr/bin/env python3
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1];src=(ROOT/"mppg_orchestrator.py").read_text();kernel=(ROOT/"mppg_recovery_kernel.py").read_text();launcher=(ROOT/"launcher.sh").read_text();tree=ast.parse(src);ktree=ast.parse(kernel)
for schema_name in ["ai_resolution.schema.json","recovery_resolution.schema.json"]:
 s=json.loads((ROOT/"schemas"/schema_name).read_text());assert s["additionalProperties"] is False and set(s["required"])==set(s["properties"])
for tr in [tree,ktree]:
 for node in ast.walk(tr):
  if isinstance(node,ast.Call):
   if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="os" and node.func.attr=="system":raise AssertionError("os.system")
   for kw in node.keywords:
    if kw.arg=="shell" and isinstance(kw.value,ast.Constant) and kw.value.value is True:raise AssertionError("shell=True")
assert "mppg_recovery_kernel.py" in launcher and "python3 -B -S -u" in launcher and "mppg_orchestrator.py" not in launcher
assert "def exception_blocker(" in src and "generated_blockers" in src and "MACHINE_ACCEPTANCE_FAILURE_REENTERED_AI_LOOP" in src
assert "def live_acceptance_test(" in src and "SYNTHETIC_CLOSED_LOOP_CANARY" in src
assert 'open("/dev/tty"' in src and 'open("/dev/tty"' in kernel
assert "ORCHESTRATOR_ERROR_EVENT_JSON=" in src and "KERNEL RECOVERY MATERIALIZATION" in kernel and "shadow_candidate" in kernel
assert "--no-hardlinks" in kernel and "py_compile.compile(sys.argv[1],cfile=sys.argv[2],doraise=True)" in src
attrs=(ROOT.parents[1]/".gitattributes").read_text()
for p in ["*.pdf binary","*.png binary","*.docx binary","*.odt binary","*.zip binary","*.sqlite binary","*.bin binary","*.qda binary"]:assert p in attrs
print("RESILIENT_ORCHESTRATOR_STATIC_TEST=PASS");print("EXCEPTION_FEEDBACK_STATIC_CANARY=PASS");print("RECOVERY_KERNEL_STATIC_CANARY=PASS");print("TTY_GATE_STATIC_CANARY=PASS")

assert "def git_ro(" in src
assert "GIT_INDEX_FILE" in src
assert "mppg-ro-index-" in src
kernel_src=(ROOT/"mppg_recovery_kernel.py").read_text(encoding="utf-8")
assert "def git_ro(" in kernel_src
assert "mppg-kernel-ro-index-" in kernel_src
assert (ROOT/"tests/test_index_shadow.py").is_file()
print("REAL_INDEX_SHADOW_STATIC_CANARY=PASS")


architecture=(ROOT.parent/"contracts/ORCHESTRATOR_ARCHITECTURE.md").read_text(encoding="utf-8")
assert "Portable host-derived candidate freeze" in architecture
assert "ZIP extraction are never an authority" in architecture
print("HOST_DERIVED_CANDIDATE_FREEZE_STATIC_CANARY=PASS")
