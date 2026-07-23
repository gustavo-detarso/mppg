from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from typing import Any
import pytest
from academic_pipeline import cli, doctor_runtime, runtime
TEST_FILE=Path(__file__).resolve(); SOFTWARE_ROOT=TEST_FILE.parents[2]; REPO_ROOT=TEST_FILE.parents[4]
LEGACY=SOFTWARE_ROOT/'app_bundle/scripts/pipeline/academic_pipeline_rc10.py'
MANIFEST=REPO_ROOT/'docs/refactor/academic-pipeline/AP-007/ap007c3_doctor_public_integration.json'
def manifest(): return json.loads(MANIFEST.read_text(encoding='utf-8'))
def test_routes():
    assert runtime.select_runtime_route(('--doctor',)) is runtime.RuntimeRoute.NATIVE_DOCTOR
    assert runtime.select_runtime_route(('--check-config',)) is runtime.RuntimeRoute.LEGACY_FALLBACK
def test_runtime_doctor_never_calls_legacy(monkeypatch):
    monkeypatch.setattr(doctor_runtime,'run_doctor_command',lambda argv:23)
    assert runtime.run(['--doctor'],legacy_runner=lambda argv:(_ for _ in ()).throw(AssertionError()))==23
def test_cli_doctor_never_calls_legacy(monkeypatch):
    monkeypatch.setattr(doctor_runtime,'run_doctor_command',lambda argv:19)
    monkeypatch.setattr(cli,'run_legacy',lambda argv:(_ for _ in ()).throw(AssertionError()))
    assert cli.main(['--doctor'])==19
def test_check_config_remains_list_fallback(monkeypatch):
    captured={}
    def fallback(argv): captured['argv']=list(argv); return 17
    monkeypatch.setattr(cli,'run_legacy',fallback); assert cli.main(['--check-config'])==17; assert captured=={'argv':['--check-config']}
def test_first_wave_precedes_doctor():
    assert runtime.select_runtime_route(('--list-layouts','--doctor')) is runtime.RuntimeRoute.NATIVE_FIRST_WAVE
def test_earlier_legacy_stage_precedes_doctor():
    assert runtime.select_runtime_route(tuple(manifest()['precedence_probe_argv'])) is runtime.RuntimeRoute.LEGACY_FALLBACK
def test_doctor_precedes_check_config():
    assert runtime.select_runtime_route(('--doctor','--check-config')) is runtime.RuntimeRoute.NATIVE_DOCTOR
def test_public_no_config_matches_historical():
    env=os.environ.copy(); env['PYTHONPATH']=str(SOFTWARE_ROOT)
    public=subprocess.run([sys.executable,'-m','academic_pipeline','--doctor'],cwd=SOFTWARE_ROOT,env=env,text=True,capture_output=True,timeout=90)
    historical=subprocess.run([sys.executable,str(LEGACY),'--doctor'],cwd=SOFTWARE_ROOT,env=env,text=True,capture_output=True,timeout=90)
    assert public.returncode==historical.returncode and public.returncode in {0,2}; assert public.stdout.rstrip()==historical.stdout.rstrip(); assert public.stderr==historical.stderr
def test_explicit_argv_preserves_process_state(monkeypatch):
    monkeypatch.setattr(doctor_runtime,'run_doctor_command',lambda argv:0); before_path=list(sys.path); before_cwd=os.getcwd(); monkeypatch.setattr(sys,'argv',['process','--check-config'])
    assert runtime.run(['--doctor'],legacy_runner=lambda argv:99)==0; assert sys.argv==['process','--check-config']; assert sys.path==before_path; assert os.getcwd()==before_cwd
def test_runtime_source_has_no_legacy_bridge():
    source=(SOFTWARE_ROOT/'academic_pipeline/runtime.py').read_text(encoding='utf-8')
    for forbidden in ('globals(','locals(','sys.path','importlib','academic_pipeline_rc10','LEGACY_MODULE_NAME'): assert forbidden not in source
def test_ap007c2_phase_local_state_is_superseded():
    c2=json.loads((REPO_ROOT/'docs/refactor/academic-pipeline/AP-007/ap007c2_doctor_native_adapter.json').read_text(encoding='utf-8'))
    assert c2['public_route']['integration_phase']=='AP-007C.3'; assert manifest()['supersedes_phase_local_route']=='AP-007C.2'
def test_manifest_records_public_integration():
    p=manifest(); assert p['phase']=='AP-007C.3'; assert p['status']=='doctor_publicly_integrated'; assert p['public_route']=={'doctor':'native_doctor','check_config':'legacy_fallback'}; assert p['semantic_exit_codes']==[0,2]
