"""Manual stage 07 CLI; read-only preflight and immutable, auditable outputs."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
import yaml

from .basal_cli import _json, _snapshot
from .confirmed_reference import load_confirmed_reference
from .reference_artifacts import sha256
from .states import STATE_NAMES
from .sweep_plan import build_cases, build_settings, plan_from_config
from .sweeps import SweepRun, run_sweeps


def _save_run(output: Path, case_id: str, variant: str, run: SweepRun,
              *, echo: bool = True) -> None:
    directory = output / "cases" / case_id / variant
    directory.mkdir(parents=True)
    record = {"status": run.status, "message": run.message, "settings": asdict(run.settings)}
    if run.result is not None:
        result = run.result
        # Raw returned values are diagnostic even if stage 07 rejects the result.
        if result.t.ndim == 1 and result.y.shape == (len(STATE_NAMES), len(result.t)):
            with (directory / "trajectory.csv").open("x", encoding="utf-8") as stream:
                np.savetxt(stream, np.column_stack((result.t, result.y.T)), delimiter=",",
                           header="tau," + ",".join(STATE_NAMES), comments="")
        else:
            record["trajectory_note"] = "Returned arrays have an invalid shape; no complete trajectory is available."
        if run.status == "completed":
            record.update(nfev=int(result.nfev), njev=int(result.njev), nlu=int(result.nlu),
                minimum_state=float(np.min(result.y)), final_time=float(result.t[-1]),
                final_state=dict(zip(STATE_NAMES, map(float, result.y[:, -1]))), metadata=result.metadata)
    else:
        record["trajectory_note"] = "Integrator returned no usable trajectory; partial internal solver states unavailable."
    _json(directory / "run.json", record)
    _json(directory / "outcomes.json", run.outcomes)
    if echo:
        print(f"{case_id}/{variant}: {run.status}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run selected stage 07 univariate sweeps from a pinned confirmed reference.")
    parser.add_argument("--config", type=Path, default=Path("config/sweeps.yaml"))
    parser.add_argument("--output-dir", type=Path, help="New child directory of the configured output_root.")
    parser.add_argument("--check-only", action="store_true", help="Validate plan and artifacts; no RHS, integration or writes.")
    args = parser.parse_args(argv)
    try:
        draft, raw, config_hash = _snapshot(args.config)
        plan = plan_from_config(draft)
        reference = load_confirmed_reference(plan.reference["preparation_dir"],
            plan.reference["confirmed_file"], plan.reference["confirmed_sha256"])
        cases, membership = build_cases(plan, reference.archive.parameters)
        settings = build_settings(plan, reference.archive)
        output_root = plan.output_root.resolve()
        results_root = Path("results").resolve()
        if output_root == results_root or not output_root.is_relative_to(results_root):
            raise ValueError("output_root must be a child of results/ in the project working directory.")
        for protected in (reference.archive.directory, reference.path.parent):
            if output_root.is_relative_to(protected) or protected.is_relative_to(output_root):
                raise ValueError("Sweep output must be separate from stage 05 and 06 artifacts.")
        if args.output_dir is not None:
            explicit = args.output_dir.resolve()
            if explicit == output_root or not explicit.is_relative_to(output_root) or explicit.exists():
                raise ValueError("--output-dir must be a new child of output_root; existing runs are immutable.")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.check_only:
        print(f"Stage 07 inputs OK: {len(cases)} unique cases; {len(membership)} axis positions; "
              f"{len(cases) * len(settings)} required runs; horizon={plan.horizon:g} tau.")
        print("Protocol: common_basal_step; full archived initial state; control intervention.")
        print(f"Fixed nominal k_syn={reference.archive.parameters.k_syn:g}, "
              f"k_Aout={reference.archive.parameters.k_Aout:g}; archived global clearance/inflammation parameters.")
        for axis, values in plan.levels.items():
            label = "multipliers" if axis == "k_BACE" else "values"
            print(f"{axis} {label}: {list(values)}")
        print(f"k_BACE absolute values: {[reference.archive.parameters.k_BACE * x for x in plan.levels['k_BACE']]}")
        print("No RHS or integration performed; no files written.")
        return 0
    started = datetime.now(timezone.utc)
    output = args.output_dir or output_root / started.strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"Cannot create a new output directory: {error}")
    (output / "sweeps.yaml").write_bytes(raw)
    for label, files in (("source_reference", reference.archive.files), ("source_confirmation", reference.files)):
        copied = output / label
        copied.mkdir()
        for name, content in files.items():
            (copied / name).write_bytes(content)
    source_root = Path(__file__).resolve().parent
    source_files = {str(path.relative_to(source_root)): path.read_bytes() for path in sorted(source_root.rglob("*.py"))}
    for name, content in source_files.items():
        snapshot = output / "implementation" / "adt" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    provenance = {"model_version": "0.3", "started_utc": started.isoformat(),
        "config_source": str(args.config.resolve()), "config_sha256": config_hash,
        "confirmed_reference_source": str(reference.path), "confirmed_reference_sha256": sha256(reference.files["reference_confirmed.json"]),
        "preparation_source": str(reference.archive.directory), "preparation_artifact_sha256": reference.archive.hashes,
        "confirmation_summary_sha256": sha256(reference.files["summary.json"]),
        "source_sha256": {name: sha256(content) for name, content in source_files.items()},
        "python": platform.python_version(), "packages": {name: version(name) for name in ("numpy", "scipy", "PyYAML")},
        "state_order": STATE_NAMES, "initial_state": dict(zip(STATE_NAMES, map(float, reference.archive.state))),
        "preparation_time": reference.archive.document["preparation_time"],
        "preparation_and_confirmation_provenance": reference.document,
        "cases": {name: asdict(case) for name, case in cases.items()}, "membership": membership,
        "settings": {name: asdict(value) for name, value in settings.items()}, "agreement": asdict(plan.agreement)}
    _json(output / "inputs.json", provenance)
    execution = run_sweeps(plan, reference, on_run=lambda name, variant, run: _save_run(output, name, variant, run))
    _json(output / "membership.json", execution.membership)
    _json(output / "outcomes.json", execution.outcomes)
    with (output / "outcomes.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(execution.outcomes[0]))
        writer.writeheader()
        writer.writerows(execution.outcomes)
    for name, report in execution.summary["cases"].items():
        _json(output / "cases" / name / "comparisons.json", report)
    summary = {**execution.summary, "provenance_file": "inputs.json",
               "provenance_sha256": sha256((output / "inputs.json").read_bytes()),
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    _json(output / "summary.json", summary)
    print(f"{summary['status']}: {len(summary['issues'])} issues\nArtifacts: {output.resolve()}")
    return 0 if summary["numerically_confirmed"] else 4
