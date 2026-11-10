"""Manual CLI and reproducibility artifacts for basal preparation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from .equilibrium import basal_parameters, preparation_from_config, prepare_reference
from .parameters import ConfigDraft, mapping
from .states import STATE_NAMES


def _snapshot(path: Path) -> tuple[ConfigDraft, bytes, str]:
    raw = path.read_bytes()
    data = mapping(yaml.safe_load(raw.decode("utf-8")), str(path))
    return ConfigDraft(path, data), raw, hashlib.sha256(raw).hexdigest()


def _json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the P1/Q2 basal reference manually.")
    parser.add_argument("--baseline", type=Path, default=Path("config/baseline.yaml"))
    parser.add_argument("--preparation", type=Path, default=Path("config/preparation.yaml"))
    parser.add_argument("--output-dir", type=Path, help="New directory under results/; existing runs are never overwritten.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs; no RHS, integration or output files.")
    args = parser.parse_args(argv)
    try:
        baseline, baseline_raw, baseline_hash = _snapshot(args.baseline)
        preparation, preparation_raw, preparation_hash = _snapshot(args.preparation)
        parameters = basal_parameters(baseline)
        settings = preparation_from_config(preparation)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        parser.error(str(error))
    if args.check_only:
        print(f"Configuration OK: 27 parameters; profile {settings.profile}; no integration performed.")
        return 0
    started = datetime.now(timezone.utc)
    output = args.output_dir or Path("results/preparation") / started.strftime("%Y%m%dT%H%M%S_%fZ")
    if not output.resolve().is_relative_to(Path("results").resolve()):
        parser.error("Output directory must be under results/ in the project working directory.")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create a new output directory: {error}")
    (output / "baseline.yaml").write_bytes(baseline_raw)
    (output / "preparation.yaml").write_bytes(preparation_raw)
    source_root = Path(__file__).resolve().parent
    source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.rglob("*.py"))}
    provenance = {
        "model_version": "0.3", "started_utc": started.isoformat(),
        "baseline_source": str(args.baseline.resolve()), "baseline_sha256": baseline_hash,
        "preparation_source": str(args.preparation.resolve()), "preparation_sha256": preparation_hash,
        "source_sha256": source_hashes, "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "PyYAML")},
        "parameters": asdict(parameters), "preparation_settings": asdict(settings),
        "subject": {"C_i": 1.}, "input": {"kind": "constant", "I_ext": 0.},
        "intervention": "control", "state_order": STATE_NAMES,
    }
    _json(output / "inputs.json", provenance)
    result = prepare_reference(parameters, settings)
    with (output / "trajectory.csv").open("x", encoding="utf-8") as stream:
        np.savetxt(stream, np.column_stack((result.t, result.y.T)), delimiter=",",
                   header="tau," + ",".join(STATE_NAMES), comments="")
    _json(output / "windows.json", [asdict(window) for window in result.windows])
    state = dict(zip(STATE_NAMES, (float(x) for x in result.y[:, -1])))
    summary = {
        "status": result.status, "message": result.message,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "last_completed_time": float(result.t[-1]), "last_completed_state": state,
        "attempted_span": result.attempted_span, "completed_windows": len(result.windows),
        "minimum_state": float(np.min(result.y)), "nfev": result.nfev,
        "njev": result.njev, "nlu": result.nlu,
        "solver_counters_scope": "returned solver windows; failed solver calls may not expose counters",
        "numerically_confirmed": False, "biological_status": "unclassified",
    }
    _json(output / "run.json", summary)
    if result.status == "converged":
        _json(output / "reference.json", {
            "label": "converged basal reference", "state_order": STATE_NAMES,
            "initial_state": state, "preparation_time": float(result.t[-1]),
            "numerically_confirmed": False, "biological_status": "unclassified",
            "provenance": provenance,
        })
    print(f"{result.status}: {result.message}\nArtifacts: {output.resolve()}")
    return {"converged": 0, "numerical_failure": 3, "budget_exhausted": 4}[result.status]
