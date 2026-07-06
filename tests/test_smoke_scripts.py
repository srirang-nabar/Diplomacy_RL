"""Smoke-run protocol (CLAUDE.md §7.2): every script in scripts/ must run
end-to-end with --smoke in CI. A script without a passing smoke test does
not exist. Grows one entry per new script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _run_smoke(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )


def test_benchmark_engine_smoke():
    r = _run_smoke("benchmark_engine.py")
    assert r.returncode == 0, r.stderr
    assert "games/sec" in r.stdout


def test_gen_bc_data_smoke():
    r = _run_smoke("gen_bc_data.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_train_bc_smoke():
    r = _run_smoke("train_bc.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout  # includes checkpoint save+reload


def test_eval_bc_smoke():
    r = _run_smoke("eval_bc.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_train_ppo_smoke():
    # the script that goes to cloud gets the strictest smoke: rollout, update,
    # checkpoint save, resume, and final load all exercised (CLAUDE.md §7.2)
    r = _run_smoke("train_ppo.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_run_tournament_smoke():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_tournament.py"), "--smoke"],
        capture_output=True, text=True, timeout=600, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_run_bop_metrics_smoke():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_bop_metrics.py"), "--smoke"],
        capture_output=True, text=True, timeout=300, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_make_report_figs_smoke():
    r = _run_smoke("make_report_figs.py")
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


def test_run_m4_grid_smoke():
    # micro-grid: 2 rows end-to-end incl. population, evals, chi2, registry
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_m4_grid.py"), "--smoke"],
        capture_output=True, text=True, timeout=300, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout
    reg = REPO / "results" / "m4_registry_smoke.json"
    assert reg.exists()
    reg.unlink()  # smoke artifacts don't accumulate
    md = REPO / "results" / "m4_registry_smoke.md"
    if md.exists():
        md.unlink()
