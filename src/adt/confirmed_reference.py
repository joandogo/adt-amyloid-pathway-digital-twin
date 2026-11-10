"""Verify the pinned stage 06 confirmation and stage 05 lineage without RHS."""

from dataclasses import dataclass
from pathlib import Path

from .parameters import finite_number, mapping
from .reference_artifacts import ArchivedReference, CORE_FILES, _object, load_reference, sha256


CONFIRMATION_FILES = (*CORE_FILES, "numerical_validation.py", "reference_artifacts.py", "validation_cli.py")
REQUIRED_CASES = ("replay", "tolerance", "max_step", "check_grid", "radau_refined",
                  "bdf_refined", "seed_fast", "seed_loaded", "hold_radau", "hold_bdf")


@dataclass
class ConfirmedReference:
    archive: ArchivedReference
    path: Path
    document: dict[str, object]
    summary: dict[str, object]
    files: dict[str, bytes]


def _passing(check: object, name: str) -> None:
    data = mapping(check, name)
    maximum = finite_number(data.get("maximum"), f"{name}.maximum")
    if data.get("passes") is not True or not 0 <= maximum <= 1:
        raise ValueError(f"Stage 06 check did not pass: {name}.")


def load_confirmed_reference(preparation_dir: str | Path, confirmed_file: str | Path,
                             expected_sha256: str) -> ConfirmedReference:
    path = Path(confirmed_file).resolve()
    raw = path.read_bytes()
    if sha256(raw) != expected_sha256:
        raise ValueError("Pinned confirmed reference SHA-256 mismatch.")
    document = _object(raw, "confirmed reference")
    archive = load_reference(preparation_dir)
    if document.get("numerically_confirmed") is not True or document.get("biological_status") != "unclassified":
        raise ValueError("Require a numerically confirmed, biologically unclassified basal reference.")
    for key in ("initial_state", "state_order", "preparation_time", "provenance"):
        if document.get(key) != archive.document.get(key):
            raise ValueError(f"Confirmed reference disagrees with stage 05: {key}.")
    confirmation = mapping(document.get("confirmation"), "confirmation")
    if confirmation.get("summary_file") != "summary.json":
        raise ValueError("Confirmation must link its sibling summary.json.")
    report_raw = (path.parent / "summary.json").read_bytes()
    if sha256(report_raw) != confirmation.get("summary_sha256"):
        raise ValueError("Stage 06 summary SHA-256 mismatch.")
    summary = _object(report_raw, "stage 06 summary")
    if (summary.get("status") != "confirmed" or summary.get("numerically_confirmed") is not True
            or summary.get("issues") != [] or summary.get("biological_status") != "unclassified"):
        raise ValueError("Stage 06 summary is not a successful confirmation without issues.")
    provenance = mapping(summary.get("provenance"), "stage 06 provenance")
    if confirmation.get("provenance") != provenance or confirmation.get("scope") != summary.get("scope"):
        raise ValueError("Stage 06 confirmation and summary provenance disagree.")
    if provenance.get("reference_artifact_sha256") != archive.hashes:
        raise ValueError("Stage 06 is linked to different or changed stage 05 artifacts.")
    hashes = mapping(provenance.get("source_sha256"), "stage 06 source hashes")
    source_root = Path(__file__).resolve().parent
    for name in CONFIRMATION_FILES:
        if hashes.get(name) != sha256((source_root / name).read_bytes()):
            raise ValueError(f"Stage 06 dependency changed: {name}; explicitly review reference lineage.")
    cases = mapping(summary.get("cases"), "stage 06 cases")
    if set(cases) != set(REQUIRED_CASES):
        raise ValueError("Stage 06 must include all ten required cases.")
    for name in REQUIRED_CASES:
        case = mapping(cases[name], name)
        if case.get("status") != "converged":
            raise ValueError(f"Stage 06 case not converged: {name}.")
        checks = mapping(case.get("checks"), f"{name}.checks")
        required = {"endpoint"}
        if name.startswith("hold_"):
            required.add("drift")
        elif not name.startswith("seed_"):
            required.add("trajectory")
        if not required <= checks.keys():
            raise ValueError(f"Stage 06 missing required comparisons: {name}.")
        for label, check in checks.items():
            _passing(check, f"{name}.{label}")
    _passing(summary.get("refined_method_comparison"), "refined methods")
    audit = summary.get("archived_window_recheck")
    count = archive.preparation.consecutive_windows
    if (not isinstance(audit, list) or len(audit) < count
            or any(mapping(item, "Q2 recheck").get("passes") is not True for item in audit[-count:])
            or finite_number(audit[-1].get("consecutive"), "Q2 consecutive windows") < count):
        raise ValueError("Stage 06 lacks the final persistent Q2 recheck.")
    return ConfirmedReference(archive, path, document, summary,
                              {"reference_confirmed.json": raw, "summary.json": report_raw})
