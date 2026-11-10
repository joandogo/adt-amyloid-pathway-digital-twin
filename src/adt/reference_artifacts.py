"""Read and verify the explicit stage 05 artifact contract, without evaluating RHS."""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml

from .equilibrium import PreparationSettings, _steps, basal_parameters, preparation_from_config
from .model import state_vector
from .parameters import ConfigDraft, ModelParameters, check_keys, finite_number, mapping
from .states import STATE_NAMES


ARTIFACT_NAMES = ("baseline.yaml", "preparation.yaml", "inputs.json", "run.json",
                  "reference.json", "trajectory.csv", "windows.json")
CORE_FILES = ("model.py", "parameters.py", "inputs.py", "interventions.py", "states.py",
              "simulate.py", "equilibrium.py", "basal_cli.py")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object(raw: bytes, name: str) -> dict[str, object]:
    def invalid(value: str) -> None:
        raise ValueError(f"{name}: nonfinite JSON constant {value}.")
    return mapping(json.loads(raw, parse_constant=invalid), name)


def _state(value: object, name: str) -> np.ndarray:
    data = mapping(value, name)
    check_keys(data, set(STATE_NAMES), name)
    if set(data) != set(STATE_NAMES):
        raise ValueError(f"{name}: all seven states are required.")
    return state_vector([finite_number(data[key], f"{name}.{key}") for key in STATE_NAMES])


@dataclass
class ArchivedReference:
    directory: Path
    parameters: ModelParameters
    preparation: PreparationSettings
    state: np.ndarray
    times: np.ndarray
    states: np.ndarray
    document: dict[str, object]
    files: dict[str, bytes]

    @property
    def hashes(self) -> dict[str, str]:
        return {name: sha256(raw) for name, raw in self.files.items()}


def load_reference(directory: str | Path) -> ArchivedReference:
    """Reject incomplete, inconsistent or stale runs before numerical work starts."""
    directory = Path(directory)
    files = {name: (directory / name).read_bytes() for name in ARTIFACT_NAMES}
    baseline = ConfigDraft(directory / "baseline.yaml", mapping(yaml.safe_load(files["baseline.yaml"]), "baseline"))
    preparation = ConfigDraft(directory / "preparation.yaml", mapping(yaml.safe_load(files["preparation.yaml"]), "preparation"))
    parameters, settings = basal_parameters(baseline), preparation_from_config(preparation)
    reference = _object(files["reference.json"], "reference")
    run, inputs = _object(files["run.json"], "run"), _object(files["inputs.json"], "inputs")
    if run.get("status") != "converged":
        raise ValueError("Stage 05 must have status converged before confirmation.")
    if reference.get("state_order") != list(STATE_NAMES) or inputs.get("state_order") != list(STATE_NAMES):
        raise ValueError("Archived state order does not match v0.3.")
    if reference.get("provenance") != inputs:
        raise ValueError("Reference provenance and inputs.json disagree.")
    if inputs.get("model_version") != "0.3" or inputs.get("parameters") != asdict(parameters):
        raise ValueError("Archived parameters/version disagree with the baseline snapshot.")
    # Convert tuples to their JSON representation, without changing any numbers.
    if inputs.get("preparation_settings") != json.loads(json.dumps(asdict(settings))):
        raise ValueError("Archived preparation settings disagree with the YAML snapshot.")
    if (inputs.get("subject") != {"C_i": 1.} or inputs.get("input") != {"kind": "constant", "I_ext": 0.}
            or inputs.get("intervention") != "control"):
        raise ValueError("Archived conditions are not the basal control reference.")
    for name, field in (("baseline.yaml", "baseline_sha256"), ("preparation.yaml", "preparation_sha256")):
        if inputs.get(field) != sha256(files[name]):
            raise ValueError(f"Snapshot hash mismatch: {name}.")
    hashes = mapping(inputs.get("source_sha256"), "source_sha256")
    root = Path(__file__).resolve().parent
    for name in CORE_FILES:
        if hashes.get(name) != sha256((root / name).read_bytes()):
            raise ValueError(f"Stage 05 source changed: {name}; review and regenerate the reference explicitly.")

    state = _state(reference.get("initial_state"), "reference.initial_state")
    stop = finite_number(reference.get("preparation_time"), "preparation_time")
    if not settings.minimum_time <= stop <= settings.maximum_time:
        raise ValueError("Archived preparation time is outside its configured budget.")
    if finite_number(run.get("last_completed_time"), "last_completed_time") != stop:
        raise ValueError("Archived run and reference times disagree.")
    if not np.array_equal(_state(run.get("last_completed_state"), "last_completed_state"), state):
        raise ValueError("Archived run and reference states disagree.")
    csv = files["trajectory.csv"].decode("utf-8")
    if not csv.splitlines() or csv.splitlines()[0] != "tau," + ",".join(STATE_NAMES):
        raise ValueError("Unexpected trajectory columns/state order.")
    matrix = np.loadtxt(io.StringIO(csv), delimiter=",", skiprows=1, ndmin=2)
    points = _steps(stop, settings.check_interval, "trajectory time")
    if matrix.shape != (points + 1, 8) or not np.all(np.isfinite(matrix)):
        raise ValueError("Incomplete or nonfinite archived trajectory.")
    times, states = matrix[:, 0], matrix[:, 1:].T
    if not np.array_equal(times, np.linspace(0., stop, points + 1)) or np.any(states < 0):
        raise ValueError("Invalid archived grid or negative states.")
    if not np.array_equal(states[:, 0], settings.initial_state) or not np.array_equal(states[:, -1], state):
        raise ValueError("Trajectory endpoints disagree with seed/reference.")
    if finite_number(run.get("minimum_state"), "minimum_state") != float(np.min(states)):
        raise ValueError("Archived minimum state disagrees with the trajectory.")
    windows = json.loads(files["windows.json"])
    count = _steps(stop, settings.window, "reference time")
    if not isinstance(windows, list) or len(windows) != count or run.get("completed_windows") != count:
        raise ValueError("Incomplete archived window history.")
    consecutive = 0
    for index, item in enumerate(windows):
        window = mapping(item, "window")
        if window.get("start") != index * settings.window or window.get("end") != (index + 1) * settings.window:
            raise ValueError("Archived window boundaries are inconsistent.")
        metrics = []
        for name in ("residual", "variation"):
            value = finite_number(window.get(name), name)
            vector = window.get(f"{name}_by_state")
            if not isinstance(vector, list) or len(vector) != 7:
                raise ValueError("Archived metrics need seven components.")
            numbers = [finite_number(x, name) for x in vector]
            if min(numbers) < 0 or value != max(numbers):
                raise ValueError("Invalid archived window metrics.")
            metrics.append(value)
        passes = max(metrics) <= 1
        consecutive = consecutive + 1 if passes else 0
        if window.get("passes") is not passes or window.get("consecutive") != consecutive:
            raise ValueError("Archived convergence flags/counters disagree with metrics.")
    if consecutive < settings.consecutive_windows:
        raise ValueError("Archived reference lacks consecutive passing windows.")
    return ArchivedReference(directory.resolve(), parameters, settings, state, times, states, reference, files)
