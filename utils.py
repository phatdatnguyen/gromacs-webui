"""Shared helpers for the GROMACS WebUI: MDP generation, GROMACS process
handling, topology merging and structure/trajectory viewer support."""

from __future__ import annotations

import importlib.util
import html
import inspect
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Any, Iterator, TypedDict

import MDAnalysis as mda
import nglview
import numpy as np
import pandas as pd
import psutil
from MDAnalysis.analysis import rms as _mda_rms
from MDAnalysis.analysis.align import rotation_matrix as _rotation_matrix
from MDAnalysis.lib.distances import minimize_vectors as _minimize_vectors
from MDAnalysis.transformations import fit_rot_trans as _fit_rot_trans
from path_security import (
    MODEL_ROOT,
    PROJECT_ROOT,
    STATIC_ROOT,
    validate_local_file_path,
    validate_working_directory,
)

# Machine learning potentials are optional and their dependency sets differ.
# Keep these imports lazy: requiring e3nn globally, for example, used to disable
# TorchANI even though e3nn is only needed while exporting MACE.
SUPPORTED_NNPOT_MODELS: tuple[str, ...] = (
    "ani1x",
    "ani2x",
    "ani2x-emle",
    "mace-small",
    "mace-medium",
    "mace-large",
    "aimnet2",
)
NNPOT_MODEL_PACKAGES: dict[str, tuple[str, ...]] = {
    "ani1x": ("torch", "torchani"),
    "ani2x": ("torch", "torchani"),
    "ani2x-emle": ("torch", "torchani", "emle"),
    "mace-small": ("torch", "mace", "e3nn"),
    "mace-medium": ("torch", "mace", "e3nn"),
    "mace-large": ("torch", "mace", "e3nn"),
    "aimnet2": ("torch", "aimnet"),
}
# Kept as a public compatibility alias.  Torch is the only dependency shared by
# every model; model-specific checks happen once a model has been selected.
NNPOT_REQUIRED_PACKAGES: tuple[str, ...] = ("torch",)
NNPOT_MODEL_BUILD_LOCK = threading.Lock()


class IonSpecies(TypedDict):
    """A monatomic species (NA, CL, CU2P) and how it should be drawn."""

    resname: str
    count: int
    element: str | None
    color: str
    recognized: bool


class HeteroSpecies(TypedDict):
    """A polyatomic non-water, non-protein residue such as a ligand."""

    resname: str
    count: int
    atoms_per_residue: int


class StructureSpecies(TypedDict):
    """Everything a viewer needs to know about what a structure contains."""

    protein_residues: int
    ions: list[IonSpecies]
    hetero: list[HeteroSpecies]
    water: list[str]


class TerminusMenu(TypedDict):
    """A single terminus menu printed by pdb2gmx -ter, with its options."""

    kind: str
    residue: str
    options: list[tuple[str, str]]


class TrajectoryViewerInfo(TypedDict):
    """Result of reducing a trajectory for the browser-side viewer."""

    frames: int
    stride: int
    total_frames: int
    n_atoms: int
    n_residues: int
    species: StructureSpecies


def atomic_write_text_file(file_path: str, content: str) -> None:
    """Replace a text file without exposing a truncated destination on failure."""
    if not isinstance(content, str):
        raise TypeError("Atomic text output must be a string.")
    destination = os.path.abspath(file_path)
    directory = os.path.dirname(destination)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".atomic_text_", suffix=".tmp",
        dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def atomic_write_dataframe_csv(file_path: str, frame: pd.DataFrame) -> None:
    """Replace a CSV only after pandas has finished writing a temporary file."""
    destination = os.path.abspath(file_path)
    directory = os.path.dirname(destination)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".atomic_csv_", suffix=".tmp",
        dir=directory)
    os.close(descriptor)
    try:
        frame.to_csv(temporary_path, index=False)
        with open(temporary_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


class XvgData(TypedDict):
    """A parsed GROMACS .xvg file: the numbers plus the labels it carries itself."""

    frame: pd.DataFrame
    title: str
    xlabel: str
    ylabel: str



def get_missing_nnpot_packages(model_name: str | None = None) -> list[str]:
    """Return absent packages for all NNPot models or for one selected model."""
    if model_name is not None and model_name not in NNPOT_MODEL_PACKAGES:
        raise ValueError(
            f"Unsupported NNPot model {model_name!r}. Choose one of: "
            + ", ".join(SUPPORTED_NNPOT_MODELS)
        )

    required_packages = (NNPOT_REQUIRED_PACKAGES if model_name is None
                         else NNPOT_MODEL_PACKAGES[model_name])
    missing = []
    for name in required_packages:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:
            # A broken or shadowed installation is as good as a missing one here.
            missing.append(name)

    return missing


def get_nnpot_unavailable_reason(model_name: str | None = None) -> str | None:
    """A message naming what to install, or None when potentials can be used."""
    missing = get_missing_nnpot_packages(model_name)
    reasons = []
    if missing:
        if model_name:
            package_reason = f"The {model_name} model is disabled"
        else:
            package_reason = "Machine learning potentials are disabled"
        reasons.append(f"{package_reason}: {', '.join(missing)} not installed. "
                       "See the Readme for the optional install steps.")

    gromacs_reason = get_gromacs_nnpot_unavailable_reason()
    if gromacs_reason is not None:
        reasons.append(gromacs_reason)
    return "\n\n".join(reasons) if reasons else None


@lru_cache(maxsize=1)
def get_gromacs_nnpot_unavailable_reason() -> str | None:
    """Explain why the ``gmx`` on PATH cannot execute TorchScript potentials."""
    executable = shutil.which("gmx")
    if executable is None:
        return "Machine learning potentials are disabled: gmx was not found on PATH."

    try:
        result = subprocess.run(
            [executable, "--version"], text=True, capture_output=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Machine learning potentials are disabled: unable to inspect gmx --version ({exc})."

    version_output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        return ("Machine learning potentials are disabled: gmx --version failed "
                f"with exit status {result.returncode}.")
    if re.search(r"^Torch support:\s*enabled\s*$", version_output, flags=re.MULTILINE | re.IGNORECASE):
        return None

    return ("Machine learning potentials are disabled: this GROMACS build reports "
            "'Torch support: disabled'. Rebuild GROMACS with GMX_NNPOT=TORCH and a "
            "LibTorch version matching the Python PyTorch used to export the model.")


GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE: str = "GMX_MMPBSA_EXECUTABLE"
# The environment the Readme tells you to build, beside the application's own.
GMX_MMPBSA_ENVIRONMENT_PATH: str = str(PROJECT_ROOT / "gmx-mmpbsa-env")

def get_gmx_mmpbsa_executable() -> str | None:
    """Where gmx_MMPBSA lives, or None when it is not installed.

    Looked for in the project's own gmx-mmpbsa-env first, so the documented
    install just works, then on PATH; an explicit environment variable overrides
    both. Never taken from a value typed into the UI: this string becomes argv[0]
    of a subprocess, and a client-supplied one would mean "run any binary on this
    machine" even though nothing is passed to a shell.
    """
    configured = os.environ.get(GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE)
    if configured:
        configured = _canonical_executable_path(configured)
        return configured if _is_executable(configured) else None

    local = _canonical_executable_path(
        os.path.join(GMX_MMPBSA_ENVIRONMENT_PATH, "bin", "gmx_MMPBSA")
    )
    if _is_executable(local):
        return local

    discovered = shutil.which("gmx_MMPBSA")
    if discovered is None:
        return None
    discovered = _canonical_executable_path(discovered)
    return discovered if _is_executable(discovered) else None


def _canonical_executable_path(path: str) -> str:
    """Return a stable absolute spelling of an executable path."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


# gmx_MMPBSA imports mpi4py unconditionally, even for a serial run, so MPI has to
# initialise before it can fall back to its serial path.
GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE: str = "I_MPI_FABRICS"
GMX_MMPBSA_DEFAULT_FABRIC: str = "shm"

def get_gmx_mmpbsa_environment(executable: str) -> dict[str, str]:
    """The environment gmx_MMPBSA needs to start out of its own installation.

    Two things it cannot do for itself:

    * Its bin goes on PATH, because mpirun is a shell script that looks up
      mpiexec.hydra by name and would otherwise not find it.
    * I_MPI_FABRICS defaults to shared memory. The Intel MPI this package pulls
      in probes for a fast fabric on startup and aborts in its OFI provider when
      there is none - which is every laptop, container and WSL install. MM-PBSA
      runs on a single node regardless, so shared memory is the right fabric,
      not merely a workaround.

    An existing value is left alone, so a cluster can set its own.
    """
    environment = dict(os.environ)
    bin_directory = os.path.dirname(os.path.abspath(executable))
    environment["PATH"] = bin_directory + os.pathsep + environment.get("PATH", "")
    environment.setdefault(GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE, GMX_MMPBSA_DEFAULT_FABRIC)

    return environment


def get_mpirun_beside(executable: str) -> str:
    """The mpirun from the same installation, so its own MPI is the one used."""
    candidate = os.path.join(os.path.dirname(os.path.abspath(executable)), "mpirun")

    return candidate if _is_executable(candidate) else "mpirun"


def get_gmx_mmpbsa_unavailable_reason() -> str | None:
    """A message naming what to install, or None when MM-PBSA can be run."""
    if get_gmx_mmpbsa_executable() is not None:
        return None

    return ("MM-PBSA is disabled: gmx_MMPBSA was not found. It pins older "
            "dependencies, so it goes in its own environment beside this one:\n\n"
            f"    conda create -p {GMX_MMPBSA_ENVIRONMENT_PATH} python=3.9\n"
            f"    conda install -p {GMX_MMPBSA_ENVIRONMENT_PATH} -c conda-forge gmx_mmpbsa\n\n"
            f"That location is found automatically. Otherwise put its bin on PATH or "
            f"point {GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE} at the binary. "
            "See the Readme.")


def is_gmx_mmpbsa_available() -> bool:
    """Whether the optional MM-PBSA support can be used."""
    return get_gmx_mmpbsa_executable() is not None


def is_nnpot_available(model_name: str | None = None) -> bool:
    """Whether the optional machine learning potential support can be used."""
    return get_nnpot_unavailable_reason(model_name) is None


def get_torchani_install_error_message(exc: Exception) -> str | None:
    """Explain a mixed TorchANI installation, or None if that is not the cause."""
    message = str(exc)
    if "PERIODIC_TABLE" not in message or "torchani.utils" not in message:
        return None

    return (
        "TorchANI could not be imported because the current environment appears "
        "to contain a mixed or stale TorchANI installation. This usually happens "
        "after installing the newer pip package over the older conda package, "
        "leaving incompatible files from both versions in site-packages.\n\n"
        "Repair the conda environment, then try generating the MDP again:\n"
        "  python -m pip uninstall -y torchani\n"
        "  python -m pip uninstall -y torchani\n"
        "  python -m pip install torchani\n"
        "  ani build-extensions\n\n"
        f"Original import error: {message}"
    )

def get_emle_install_error_message(model_name: str, exc: Exception) -> str | None:
    """Explain a broken EMLE installation, or None if that is not the cause."""
    if model_name != "ani2x-emle":
        return None

    message = str(exc)
    if "pygit2" in message:
        return (
            "EMLE could not fetch its model resources because the Python "
            "package 'pygit2' is missing.\n\n"
            "Install it in the active environment, then regenerate the MDP:\n"
            "  python -m pip install pygit2\n\n"
            f"Original import error: {message}"
        )
    if "SpeciesEnergies" in message and "torchani" in message:
        return (
            "EMLE could not import TorchANI compatibility classes. The wrapper "
            "adds a TorchANI 2.8 compatibility shim, but this EMLE/TorchANI "
            "combination still appears incompatible.\n\n"
            "Try reinstalling EMLE after TorchANI, or use the TorchANI version "
            "recommended by your EMLE checkout.\n\n"
            f"Original import error: {message}"
        )

    return None

def get_nnpot_model_load_error_message(exc: Exception) -> str | None:
    """Explain a cached model needing a rebuild, or None if that is not the cause."""
    message = str(exc)
    if "torch.classes.cuaev.CuaevComputer" not in message:
        return None

    return (
        "The cached ANI model was scripted with TorchANI's optional cuAEV "
        "extension, but that custom class is not available when the model is "
        "loaded. Rebuild the cached ANI model with the pure PyTorch AEV "
        "strategy.\n\n"
        f"Original model load error: {message}"
    )

def get_expected_nnpot_model_config(model_name: str) -> str:
    """Return the config fingerprint a cached model file must carry to be reusable."""
    if model_name in ["ani1x", "ani2x"]:
        return f"{model_name}|torchani|pyaev|adaptive|neutral-charge-check|nonmutating-box-v3"
    if model_name == "ani2x-emle":
        return f"{model_name}|emle|electrostatic-mm-forces|runtime-charge|pyaev-v4"
    if model_name.startswith("mace-"):
        return f"{model_name}|mace|gromacs-pairs-0.5nm|neutral-charge-check|energy-only-v6"
    if model_name == "aimnet2":
        return f"{model_name}|aimnet|traced-runtime-charge-box-pbc-device-float64-v6"
    return model_name

def is_cached_nnpot_model_usable(model_name: str, modelfile_path: str) -> bool:
    """Report whether the cached model matches this build, moving it aside if not."""
    import torch

    def quarantine(reason: str) -> bool:
        backup_path = modelfile_path + ".invalid"
        os.replace(modelfile_path, backup_path)
        print(f"Moved {reason} cached NNPot model to {backup_path}.")
        return False

    extra_files = {"nnpot_model_config": ""}
    try:
        torch.jit.load(modelfile_path, map_location="cpu", _extra_files=extra_files)
        cached_config = extra_files["nnpot_model_config"]
        if isinstance(cached_config, bytes):
            cached_config = cached_config.decode()
        if cached_config != get_expected_nnpot_model_config(model_name):
            return quarantine("outdated")
        return True
    except RuntimeError as exc:
        if get_nnpot_model_load_error_message(exc) is not None:
            return quarantine("unusable")
        # Interrupted/partial atomic-cache migrations and disk damage normally
        # surface from torch.jit.load as one of these archive-reader failures.
        # Treat only recognisable serialization damage as a stale cache; other
        # runtime failures may describe a missing custom operator and must remain
        # visible instead of triggering an expensive rebuild loop.
        message = str(exc).lower()
        corrupt_markers = (
            "pytorchstreamreader failed",
            "failed finding central directory",
            "file is not a zip file",
            "unexpected end of file",
            "failed locating file",
            "constants.pkl",
            "data.pkl",
        )
        if any(marker in message for marker in corrupt_markers):
            return quarantine("corrupt")
        raise

def checkExtensions() -> dict[str, str]:
    """Collect loaded Torch extension libraries to embed alongside a saved model."""
    import torch

    ext_lib = []
    for lib in torch.ops.loaded_libraries:
        if lib:
            ext_lib.append(lib)
    ext_lib = ":".join(ext_lib)
    print("Loaded extension libraries: ", ext_lib)
    extra_files = {}
    if ext_lib:
        extra_files['extension_libs'] = ext_lib
    return extra_files

def trace_aimnet2_model(model: torch.nn.Module) -> torch.jit.ScriptModule:
    """Trace AIMNet2 with representative inputs, since it cannot be scripted."""
    import torch

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0], [-0.02399, 0.0927, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.int64, device=device)
    nnp_charge = torch.tensor(0.0, dtype=torch.float64, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device)
    pbc = torch.tensor([True, True, True], device=device)
    return torch.jit.trace(
        model,
        (positions, atomic_numbers, nnp_charge, cell, pbc),
        strict=False,
        check_trace=False,
    )

def download_nnpot_model(model_name: str) -> str:
    """Build or reuse a selected neural-network potential and return its path.

    Model names are validated before becoming part of a filesystem path.  The
    lock prevents two Gradio sessions from exporting the same (large) model at
    once, while the atomic save protects readers in other server processes.
    """
    reason = get_nnpot_unavailable_reason(model_name)
    if reason is not None:
        raise RuntimeError(reason)

    with NNPOT_MODEL_BUILD_LOCK:
        return _download_nnpot_model_locked(model_name)


def _download_nnpot_model_locked(model_name: str) -> str:
    """Implementation of :func:`download_nnpot_model`, called under its lock."""

    import torch
    from nnpot_models import (
        GmxAIMNet2Model,
        GmxANI1xModel,
        GmxANI2xEMLEModel,
        GmxANI2xModel,
        GmxMACEModel,
    )

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(MODEL_ROOT / "warp-cache"))
    os.environ.setdefault("AIMNET_CACHE_DIR", str(MODEL_ROOT / "aimnet-cache"))
    # Absolute: this path is written into the MDP as nnpot-modelfile and resolved
    # by mdrun, which runs from the job directory rather than the repository root.
    modelfile_path = str(MODEL_ROOT / f"{model_name}.pt")
    is_ani_model = model_name in ["ani1x", "ani2x"]
    is_emle_model = model_name == "ani2x-emle"
    
    if os.path.exists(modelfile_path):
        if not is_cached_nnpot_model_usable(model_name, modelfile_path):
            print(f"Rebuilding {model_name}.")
        else:
            return modelfile_path

    if os.path.exists(modelfile_path):
        return modelfile_path

    # Download the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        if is_ani_model or model_name == "ani2x-emle":
            os.environ["TORCHANI_DISABLE_EXTENSIONS"] = "1"
        if model_name=="ani1x":
            model = GmxANI1xModel(device)
        elif model_name=="ani2x":
            model = GmxANI2xModel(device)
        elif model_name=="aimnet2":
            model = GmxAIMNet2Model(device)
        elif model_name=="ani2x-emle":
            model = GmxANI2xEMLEModel(device)
        else:
            model_size = model_name.split('-')[1]
            model = GmxMACEModel(model_size, device)
    except ImportError as exc:
        torchani_message = get_torchani_install_error_message(exc)
        if torchani_message is not None:
            raise RuntimeError(torchani_message) from exc
        emle_message = get_emle_install_error_message(model_name, exc)
        if emle_message is not None:
            raise RuntimeError(emle_message) from exc
        raise
    
    # Save the model
    extensions = checkExtensions()
    extensions["nnpot_model_config"] = get_expected_nnpot_model_config(model_name)
    if model_name == "aimnet2":
        scripted_model = trace_aimnet2_model(model)
    elif is_emle_model or not model_name.startswith("mace-"):
        scripted_model = torch.jit.script(model)
    else:
        # MACE uses e3nn's scripting adapter; other models do not require e3nn.
        from e3nn.util.jit import script
        scripted_model = script(model)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{model_name}.", suffix=".pt.tmp", dir=MODEL_ROOT)
    os.close(file_descriptor)
    try:
        scripted_model.save(temporary_path, _extra_files=extensions)
        os.replace(temporary_path, modelfile_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"Saved wrapped model to {modelfile_path}.")
    
    return modelfile_path

MAX_CAPTURED_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
_COMMAND_OUTPUT_TRUNCATION_MARKER = (
    b"\n... command output truncated by GROMACS WebUI ...\n")


class _BoundedCommandOutput:
    """A fixed-memory head/tail capture that still drains the complete pipe."""

    def __init__(self, limit: int = MAX_CAPTURED_COMMAND_OUTPUT_BYTES) -> None:
        self.limit = limit
        self.head_limit = limit // 4
        self.tail_limit = limit - self.head_limit - len(
            _COMMAND_OUTPUT_TRUNCATION_MARKER)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def append(self, value: str | bytes) -> None:
        payload = value.encode("utf-8", errors="replace") \
            if isinstance(value, str) else value
        self.total += len(payload)
        missing_head = self.head_limit - len(self.head)
        if missing_head > 0:
            self.head.extend(payload[:missing_head])
            payload = payload[missing_head:]
        if payload:
            self.tail.extend(payload)
            if len(self.tail) > self.tail_limit:
                del self.tail[:-self.tail_limit]

    def text(self) -> str:
        if self.total <= self.head_limit + self.tail_limit:
            payload = bytes(self.head + self.tail)
        else:
            payload = bytes(
                self.head + _COMMAND_OUTPUT_TRUNCATION_MARKER + self.tail)
        return payload.decode("utf-8", errors="replace")


def _drain_command_stream(stream: io.TextIOBase,
                          output: _BoundedCommandOutput) -> None:
    """Continuously drain one child stream without retaining unbounded data."""
    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            return
        output.append(chunk)


def _read_bounded_command_output(handle: Any) -> str:
    """Read the beginning and end of a captured stream within a memory cap."""
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(0)
    if size <= MAX_CAPTURED_COMMAND_OUTPUT_BYTES:
        payload = handle.read()
    else:
        marker = (b"\n... command output truncated by GROMACS WebUI ...\n")
        head_size = MAX_CAPTURED_COMMAND_OUTPUT_BYTES // 4
        tail_size = MAX_CAPTURED_COMMAND_OUTPUT_BYTES - head_size - len(marker)
        head = handle.read(head_size)
        handle.seek(-tail_size, os.SEEK_END)
        payload = head + marker + handle.read(tail_size)
    return payload.decode("utf-8", errors="replace")


def run_managed_command(cmd: Sequence[str], cwd: str | None = None,
                        stdin_input: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a registered child with bounded output, regardless of its exit code."""
    execution_directory = os.path.realpath(cwd or os.getcwd())
    job_key = get_process_job_key(
        execution_directory, f".checked-command-{uuid.uuid4().hex}")
    claimed, _ = reserve_process_job(job_key)
    if not claimed:
        raise WorkingDirectoryBusyError(
            "The working directory is busy with another file operation.")

    proc: subprocess.Popen[str] | None = None
    try:
        # Pipes are drained into fixed-size head/tail rings. Unlike a spooled
        # temporary file, output volume can no longer consume unbounded /tmp.
        proc = subprocess.Popen(
            list(cmd), cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=(os.name == "posix"))
        if os.name == "posix":
            # start_new_session makes this invariant true even if the
            # leader exits before getpgid() can observe it.
            try:
                setattr(proc, "_gromacs_webui_process_group", proc.pid)
            except (AttributeError, TypeError):
                pass
        capture_owned_process_group(proc)
        try:
            register_process_job(
                job_key, proc, job_name=" ".join(str(part) for part in cmd[:2]),
                working_directory_path=execution_directory)
        except BaseException:
            stop_process_gracefully(proc, timeout=0)
            # Registration can fail while our original slot is still the
            # reserved sentinel (for example, if an observer raises during the
            # hand-off).  The identity-based cleanup in ``finally`` deliberately
            # cannot remove that sentinel because ``proc`` was never installed.
            # Release only the still-reserved slot here; if another process has
            # replaced it, ``release_process_job`` leaves that process alone.
            release_process_job(job_key)
            raise

        stdout_capture = _BoundedCommandOutput()
        stderr_capture = _BoundedCommandOutput()
        real_pipes = (isinstance(getattr(proc, "stdout", None), io.TextIOBase)
                      and isinstance(getattr(proc, "stderr", None), io.TextIOBase))
        try:
            if real_pipes:
                drain_threads = [
                    threading.Thread(
                        target=_drain_command_stream,
                        args=(proc.stdout, stdout_capture), daemon=True),
                    threading.Thread(
                        target=_drain_command_stream,
                        args=(proc.stderr, stderr_capture), daemon=True),
                ]
                for thread in drain_threads:
                    thread.start()
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(stdin_input or "")
                        proc.stdin.close()
                except BrokenPipeError:
                    pass
                proc.wait()
                stop_process_gracefully(
                    proc, timeout=1.0, mark_stopped_by_user=False)
                for thread in drain_threads:
                    thread.join(timeout=2)
                stdout, stderr = stdout_capture.text(), stderr_capture.text()
            else:
                # Compatibility for small Popen adapters used by tests and
                # integrations; their returned output is bounded immediately.
                communicated_stdout, communicated_stderr = proc.communicate(
                    input=stdin_input or "")
                if isinstance(communicated_stdout, (str, bytes)):
                    stdout_capture.append(communicated_stdout)
                if isinstance(communicated_stderr, (str, bytes)):
                    stderr_capture.append(communicated_stderr)
                stdout, stderr = stdout_capture.text(), stderr_capture.text()
        except BaseException:
            stop_process_gracefully(proc)
            raise
        process = subprocess.CompletedProcess(
            list(cmd), proc.returncode, stdout=stdout, stderr=stderr)
    finally:
        if proc is not None:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(proc, stream_name, None)
                if isinstance(stream, io.IOBase) and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        release_process_job(job_key, proc)

    return process


def run_checked_command(cmd: Sequence[str], cwd: str | None = None,
                        stdin_input: str | None = None,
                        error_lines: int = 25) -> subprocess.CompletedProcess[str]:
    """Run a command to completion, raising an Exception that carries its stderr.

    GROMACS writes its diagnostics ("Fatal error", missing atoms, mismatched
    coordinate counts) to stderr. Without capturing it, a failure surfaces in the
    UI as nothing more than 'returned non-zero exit status 1'."""
    process = run_managed_command(cmd, cwd=cwd, stdin_input=stdin_input)
    if process.returncode != 0:
        stderr_output = process.stderr or ""
        output = stderr_output + "\n" + (process.stdout or "")
        # GROMACS reserves stderr for banners and diagnostics while several
        # tools put routine progress on stdout.  Once stderr contains a known
        # error marker, do not append that progress after the fatal block.
        diagnostic_output = (
            stderr_output if any(
                marker in stderr_output
                for marker in ("Fatal error", "Error in user input",
                               "Inconsistency in user input"))
            else output
        )
        lines = [line for line in diagnostic_output.splitlines() if line.strip()]

        # GROMACS prints a version banner before the diagnostic, so start the
        # message at the error block itself and fall back to the tail otherwise.
        marker_index = None
        for index, line in enumerate(lines):
            if line.lstrip().startswith(("Fatal error", "Error in user input", "Inconsistency in user input")):
                marker_index = index

        too_many_warnings = "too many warnings" in output.lower()
        if marker_index is None:
            detail_lines = lines[-error_lines:]
        else:
            detail_lines = [line for line in lines[marker_index:]
                            if not line.startswith("---")
                            and "troubleshooting" not in line
                            and "manual.gromacs.org" not in line][:error_lines]

        # A failed grompp normally puts the useful warning well before its final
        # ``Fatal error: Too many warnings`` block.  Showing only the fatal tail
        # tells the user to inspect a warning that the UI then hides.  Prefix the
        # bounded warning blocks while retaining the fatal diagnostic.
        if too_many_warnings:
            warning_lines: list[str] = []
            for block in _extract_gromacs_warning_blocks(output):
                warning_lines.extend(
                    line for line in block.splitlines() if line.strip())
            if warning_lines:
                warning_budget = max(1, error_lines // 2)
                detail_lines = (warning_lines[:warning_budget]
                                + detail_lines[:error_lines - warning_budget])

        numbered_errors = _extract_gromacs_error_blocks(diagnostic_output)
        if numbered_errors:
            input_error_lines = [
                line for block in numbered_errors for line in block.splitlines()
                if line.strip()
            ]
            input_error_budget = max(1, error_lines // 2)
            detail_lines = (input_error_lines[:input_error_budget]
                            + detail_lines[:error_lines - input_error_budget])

        detail = "\n".join(detail_lines) if detail_lines else "no output captured"
        raise Exception(f"{os.path.basename(cmd[0])} {cmd[1] if len(cmd) > 1 else ''} failed "
                        f"(exit status {process.returncode}):\n{detail}")

    return process


_GROMACS_WARNING_HEADER_RE = re.compile(
    r"(?m)^\s*WARNING\s+\d+\b.*$")


def _extract_gromacs_warning_blocks(output: str) -> list[str]:
    """Extract numbered GROMACS warning paragraphs from mixed stderr output."""
    blocks: list[str] = []
    for match in _GROMACS_WARNING_HEADER_RE.finditer(output):
        paragraph_end = output.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(output)
        blocks.append(output[match.start():paragraph_end].strip())
    return blocks


_GROMACS_ERROR_HEADER_RE = re.compile(r"(?m)^\s*ERROR\s+\d+\b.*$")


def _extract_gromacs_error_blocks(output: str) -> list[str]:
    """Extract numbered grompp input-error paragraphs before its fatal footer."""
    blocks: list[str] = []
    for match in _GROMACS_ERROR_HEADER_RE.finditer(output):
        paragraph_end = output.find("\n\n", match.end())
        if paragraph_end < 0:
            paragraph_end = len(output)
        blocks.append(output[match.start():paragraph_end].strip())
    return blocks

# "Group     1 (        Protein) has 45 elements" — the menu every interactive gmx
# analysis tool prints before asking which group to use.
_GMX_GROUP_LINE = re.compile(r'Group\s+(\d+)\s*\(\s*([^)]+?)\s*\)')

def find_gmx_group_number(gmx_output: str, group_name: str) -> str | None:
    """The index gmx offered for a named group, or None when it offered no such group.

    Group numbering is not fixed: it depends on the force field and on what the
    system contains. Every caller must look the number up in the tool's own menu
    rather than assuming a well-known index."""
    for number, name in _GMX_GROUP_LINE.findall(gmx_output):
        if name == group_name:
            return number

    return None

def parse_gmx_groups(gmx_output: str) -> dict[str, str]:
    """Every group gmx offered, as {name: index}, first occurrence winning."""
    groups: dict[str, str] = {}
    for number, name in _GMX_GROUP_LINE.findall(gmx_output):
        groups.setdefault(name, number)

    return groups

# "  13 UNK                 :    74 atoms" - the listing gmx make_ndx prints. Note
# this is a different shape from the "Group 13 ( UNK )" the analysis tools print.
_GMX_MAKE_NDX_LINE = re.compile(r'^\s*(\d+)\s+(\S.*?)\s*:\s*(\d+)\s+atoms\s*$')

# Groups gmx builds for every system, so none of them identifies a ligand.
GMX_STANDARD_INDEX_GROUPS: frozenset[str] = frozenset({
    "System", "Protein", "Protein-H", "C-alpha", "Backbone", "MainChain",
    "MainChain+Cb", "MainChain+H", "SideChain", "SideChain-H", "Prot-Masses",
    "non-Protein", "Other", "Water", "SOL", "non-Water", "Ion", "Water_and_ions",
    "DNA", "RNA",
})
GMX_COMMON_ION_NAMES: frozenset[str] = frozenset({
    "NA", "CL", "K", "MG", "ZN", "CA", "NA+", "CL-", "CU", "FE",
})

def list_gmx_index_groups(structure_file_name: str,
                          working_directory_path: str) -> list[tuple[str, int]]:
    """The default index groups gmx builds for a structure, as (name, atom count).

    Uses gmx make_ndx because it works with any tpr, including versions newer
    than the MDAnalysis parser understands.
    """
    output_file_name = ".probe_make_ndx.ndx"
    try:
        completed = run_checked_command(
            ["gmx", "make_ndx", "-f", structure_file_name, "-o", output_file_name],
            cwd=working_directory_path, stdin_input="q\n")
    finally:
        try:
            os.remove(os.path.join(working_directory_path, output_file_name))
        except OSError:
            pass

    groups: list[tuple[str, int]] = []
    for line in (completed.stderr + completed.stdout).splitlines():
        match = _GMX_MAKE_NDX_LINE.match(line)
        if match:
            groups.append((match.group(2), int(match.group(3))))

    return groups

def describe_selection_candidates(structure_file_name: str,
                                  working_directory_path: str) -> str:
    """A sentence naming what a selection could have meant, or "" if unavailable.

    A selection that matches nothing is nearly always a residue name that differs
    from the one assumed - a job set up before the ligand was normalised to LIG,
    for instance - so the fix is to say what the structure does contain.
    """
    try:
        groups = list_gmx_index_groups(structure_file_name, working_directory_path)
    except Exception:
        # Only ever used to enrich another error; never replace it with this one.
        return ""

    candidates = [f"{name} ({count} atoms)" for name, count in groups
                  if name not in GMX_STANDARD_INDEX_GROUPS
                  and name.upper() not in GMX_COMMON_ION_NAMES]
    if not candidates:
        return ""

    return (f"{structure_file_name} contains {', '.join(candidates)}. "
            f"If one of those is the ligand, select it by name, for example "
            f"'resname {candidates[0].split(' ')[0]}'.")

def probe_gmx_groups(cmd: Sequence[str], cwd: str | None = None) -> dict[str, str]:
    """Run a command far enough to capture the group menu it prints, then discard it.

    The tool is answered immediately and its output is thrown away; only the menu
    matters. Answers go in up front because the menu lands on a block-buffered pipe,
    so reading before replying would deadlock."""
    probe = run_managed_command(cmd, cwd=cwd, stdin_input="\n" * 8)

    # trjconv and the analysis tools print the menu on stderr, but not all of them
    # agree on that, so search both streams.
    return parse_gmx_groups(probe.stderr + probe.stdout)


def get_gmx_group_input(cmd: Sequence[str], group_names: Sequence[str],
                        working_directory_path: str) -> str:
    """Resolve interactive GROMACS group names and return their stdin answers.

    Default group numbers depend on the structure. A throwaway probe reads the
    exact menu for this command so callers never rely on values such as Protein
    being group 1 or Backbone being group 4.
    """
    working_directory_path = validate_working_directory(working_directory_path)
    probe_cmd = list(cmd)
    probe_output_path: str | None = None
    if "-o" in probe_cmd:
        output_index = probe_cmd.index("-o") + 1
        suffix = os.path.splitext(str(probe_cmd[output_index]))[1]
        descriptor, probe_output_path = tempfile.mkstemp(
            prefix=".probe_groups_", suffix=suffix, dir=working_directory_path)
        os.close(descriptor)
        os.remove(probe_output_path)
        probe_cmd[output_index] = probe_output_path

    try:
        groups = probe_gmx_groups(probe_cmd, cwd=working_directory_path)
    finally:
        if probe_output_path is not None:
            try:
                os.remove(probe_output_path)
            except OSError:
                pass

    missing = [name for name in group_names if name not in groups]
    if missing:
        available = ", ".join(groups) if groups else "none"
        raise Exception(
            f"GROMACS did not offer the required group(s): {', '.join(missing)}. "
            f"Available groups: {available}.")

    return "".join(f"{groups[name]}\n" for name in group_names)

def _owned_process_group(proc: subprocess.Popen[str]) -> int | None:
    """Return a child's private POSIX process group, never the server's group."""
    if os.name != "posix" or not hasattr(os, "getpgid") or not hasattr(os, "killpg"):
        return None
    captured = getattr(proc, "_gromacs_webui_process_group", None)
    pid_value = getattr(proc, "pid", None)
    # Popen.pid is always a plain integer.  Coercing arbitrary objects here is
    # unsafe: mocks and adapters commonly implement ``int(value)`` as 1, which
    # could make shutdown signal the init/system process group and also makes a
    # finished test double appear to own live descendants forever.
    if (not isinstance(pid_value, int) or isinstance(pid_value, bool)
            or pid_value <= 0):
        return None
    pid = pid_value
    if (isinstance(captured, int) and not isinstance(captured, bool)
            and captured == pid):
        return captured
    try:
        process_group = os.getpgid(pid)
    except (OSError, TypeError, ValueError):
        return None
    # start_new_session=True and process_group=0 both make the child its group
    # leader. Never signal a group it does not own: that could include this server.
    return process_group if process_group == pid else None


def capture_owned_process_group(proc: subprocess.Popen[Any]) -> int | None:
    """Remember a private process group before a short-lived leader can exit."""
    process_group = _owned_process_group(proc)
    if process_group is not None:
        try:
            setattr(proc, "_gromacs_webui_process_group", process_group)
        except Exception:
            pass
    return process_group


def _process_group_exists(process_group: int) -> bool:
    """Whether a captured POSIX process group still has any members."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # This should not occur for our own children, but it still proves that
        # the group exists and must not be treated as safely stopped.
        return True
    return True


def _process_group_has_live_members(process_group: int) -> bool:
    """Ignore already-dead zombies when checking a group on Linux."""
    proc_root = "/proc"
    if os.path.isdir(proc_root):
        try:
            entries = os.scandir(proc_root)
        except OSError:
            return _process_group_exists(process_group)
        with entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    with open(os.path.join(entry.path, "stat")) as handle:
                        stat_line = handle.read()
                    # The command name in field 2 can contain spaces and
                    # parentheses, so split only after its final ')'.
                    fields = stat_line[stat_line.rfind(")") + 2:].split()
                    state = fields[0]
                    member_group = int(fields[2])
                except (FileNotFoundError, OSError, IndexError, ValueError):
                    continue
                if member_group == process_group and state not in {"Z", "X"}:
                    return True
        return False
    return _process_group_exists(process_group)


def stop_process_gracefully(proc: subprocess.Popen[str] | None, timeout: float = 15,
                            mark_stopped_by_user: bool = True) -> None:
    """Ask a run to stop, only killing it if it ignores the request.

    mdrun handles SIGTERM by finishing the current step, writing a checkpoint and
    a confout structure; SIGKILL would discard everything since the last
    checkpoint was written."""
    if proc is None:
        return
    if isinstance(timeout, bool):
        raise ValueError("Process stop timeout must be finite and non-negative.")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(
            "Process stop timeout must be finite and non-negative.") from None
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Process stop timeout must be finite and non-negative.")
    deadline = time.monotonic() + timeout

    process_group = _owned_process_group(proc)
    leader_running = proc.poll() is None
    if not leader_running and (process_group is None
                               or not _process_group_has_live_members(process_group)):
        return

    # Other browser sessions may be observing this same registered process.  The
    # marker lets their watcher distinguish an intentional stop from a crash.
    if mark_stopped_by_user:
        try:
            setattr(proc, "_gromacs_webui_stopped_by_user", True)
        except Exception:
            pass

    if process_group is not None:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            # The group may have disappeared between poll()/getpgid()/killpg().
            # Fall back to the child itself, but treat the same exit race as a
            # successful stop rather than leaking an exception into the UI.
            try:
                proc.terminate()
            except ProcessLookupError:
                return
            process_group = None
    else:
        if not leader_running:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    leader_exited = not leader_running
    group_killed = False
    try:
        if leader_running:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            leader_exited = True
    except subprocess.TimeoutExpired:
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGKILL)
                group_killed = True
            except ProcessLookupError:
                return
            except OSError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    return
        else:
            try:
                proc.kill()
            except ProcessLookupError:
                return
        # Reap it, otherwise the caller still sees a live process for a moment
        # and the child lingers as a zombie.
        proc.wait()

    # A session leader can exit promptly while an MPI/helper descendant ignores
    # SIGTERM.  Waiting only for ``proc`` would then release the job registry
    # while that orphan was still writing.  Keep ownership through the same
    # deadline and kill the captured group if any member remains.  A process
    # group ID cannot be reused while an original member still holds it.
    if leader_exited and process_group is not None:
        while _process_group_has_live_members(process_group):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(process_group, signal.SIGKILL)
                    group_killed = True
                except ProcessLookupError:
                    pass
                break
            time.sleep(min(0.05, remaining))

    # Signal delivery is asynchronous.  Give the kernel a short, bounded chance
    # to move killed descendants out of a runnable state before the registry is
    # released.  Zombies are harmless and may only be reaped by their new parent.
    if group_killed and timeout > 0:
        settle_deadline = time.monotonic() + min(0.1, max(0.02, timeout * 0.1))
        while (_process_group_has_live_members(process_group)
               and time.monotonic() < settle_deadline):
            time.sleep(0.005)

def get_mdrun_hardware_options(use_gpu: bool, mpi_rank: int) -> list[str]:
    """Task assignment for a restrained equilibration run.

    With the GPU asked for, offload nonbonded and PME: they carry almost all of
    the cost, while -bonded gpu and -update gpu are refused outright when
    position restraints are present, and GPU PME needs a single rank.

    Without it, name the CPU for every task rather than passing nothing. Each
    -nb/-pme/-bonded option defaults to "auto", which resolves to a detected
    GPU, so on a CUDA-enabled build silence still lands the run on the GPU."""
    if not use_gpu:
        return get_cpu_only_mdrun_options()

    options = ["-nb", "gpu"]
    if int(mpi_rank) == 1:
        options.extend(["-pme", "gpu"])

    return options

def get_cpu_only_mdrun_options() -> list[str]:
    """mdrun flags that pin a run to the CPU, whatever hardware is present.

    Energy minimisation needs them: GROMACS has no GPU PME implementation for
    the minimisers, and mdrun offloads to a detected GPU by default, so leaving
    the choice on "auto" makes minimisation fail on a GPU machine."""
    return ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu"]

class ProcessStateDict(dict):
    """Server-side state for one long-running UI action.

    ``gr.State`` deep-copies its initial value for each browser session.  A lock
    cannot safely be copied, and a ``Popen`` object only has meaning in the
    server process, so a new session deliberately starts with a clean state.
    The process registry below reconnects that clean state to a run which is
    still active for the same output path.
    """
    def __init__(self) -> None:
        super().__init__({
            "proc": None,
            "running": False,
            "lock": threading.Lock(),
            "job_key": None,
            "job_name": None,
            "working_directory": None,
            "returncode": None,
            "completion_status": None,
            "completion_color": None,
            "completion_pending": False,
            "failure_hint": None,
        })

    def __deepcopy__(self, memo: dict[int, Any]) -> ProcessStateDict:
        return ProcessStateDict()


# A browser-local gr.State is not enough to guard an output file.  Reloading the
# page (or opening a second session) creates another state and used to allow a
# second mdrun to write the same trajectory/checkpoint concurrently.  Registry
# entries are process-local by design: all jobs are children of this server, and
# are gone if the server itself is restarted.
_PROCESS_REGISTRY_LOCK = threading.Lock()
_PROCESS_REGISTRY: dict[str, object] = {}
_PROCESS_REGISTRY_METADATA: dict[str, tuple[str, str, str | None]] = {}
_PROCESS_RESERVATION_OWNERS: dict[str, int] = {}
_PROCESS_SLOT_RESERVED = object()
# Compute-intensive jobs are serialized per working directory above, but jobs
# in different directories still compete for the same host.  Keep their CPU and
# GPU admissions under the registry lock so two simultaneous launch callbacks
# cannot both observe the same capacity as free.
_PROCESS_RESOURCE_RESERVATIONS: dict[str, tuple[int, bool]] = {}
# File-manager callbacks briefly hold a directory-wide maintenance lease.  The
# lease and process-output reservations share one lock, which closes the race in
# which a callback checks that a job is idle just before another session starts
# an mdrun in that same directory.
# Canonical directory -> (owning thread id, re-entrancy depth).  Mutation
# callbacks often call a managed GROMACS helper or a publication helper which
# takes the same lease again.  Tracking ownership lets that one callback nest
# safely without admitting a writer from another Gradio worker.
_DIRECTORY_MAINTENANCE_RESERVATIONS: dict[str, tuple[int, int]] = {}
# Analysis/view callbacks take a separate exclusive read lease.  It blocks
# external writers while allowing that callback's own GROMACS helper process.
_DIRECTORY_READ_RESERVATIONS: dict[str, int] = {}


class WorkingDirectoryBusyError(RuntimeError):
    """Raised when a file mutation would overlap another writer in a job."""


class ResourceAdmissionError(RuntimeError):
    """Raised when a compute job would exceed the server's CPU/GPU budget."""


def _canonical_process_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _path_is_within_directory(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath((directory, path)) == directory
    except ValueError:
        # Different drives on Windows cannot overlap.
        return False


def _directories_overlap(first: str, second: str) -> bool:
    return (_path_is_within_directory(first, second)
            or _path_is_within_directory(second, first))


def _read_optional_system_text(path: str) -> str | None:
    """Read a small procfs/cgroup control file, or report it unavailable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeError):
        return None


def _decode_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for path fields in /proc/*/mountinfo."""
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _cgroup_directory(cgroup_path: str, mount_root: str,
                      mount_point: str) -> str | None:
    """Map a kernel cgroup path into the matching mounted filesystem path."""
    cgroup_path = os.path.normpath(cgroup_path)
    mount_root = os.path.normpath(mount_root)
    mount_point = os.path.normpath(mount_point)
    if cgroup_path == mount_root:
        relative = ""
    elif cgroup_path.startswith(mount_root.rstrip(os.sep) + os.sep):
        relative = cgroup_path[len(mount_root):].lstrip(os.sep)
    else:
        return None
    directory = os.path.normpath(os.path.join(mount_point, relative))
    try:
        if os.path.commonpath((directory, mount_point)) != mount_point:
            return None
    except ValueError:
        return None
    return directory


def _finite_cgroup_cpu_quotas(directory: str, mount_point: str,
                              version: int) -> list[int]:
    """Return conservative whole-CPU limits on this cgroup and its parents."""
    quotas: list[int] = []
    current = os.path.normpath(directory)
    mount_point = os.path.normpath(mount_point)
    while True:
        try:
            if version == 2:
                content = _read_optional_system_text(
                    os.path.join(current, "cpu.max"))
                fields = content.split() if content is not None else []
                if len(fields) >= 2 and fields[0] != "max":
                    quota, period = int(fields[0]), int(fields[1])
                    if quota > 0 and period > 0:
                        quotas.append(max(1, quota // period))
            else:
                quota_text = _read_optional_system_text(
                    os.path.join(current, "cpu.cfs_quota_us"))
                period_text = _read_optional_system_text(
                    os.path.join(current, "cpu.cfs_period_us"))
                if quota_text is not None and period_text is not None:
                    quota, period = int(quota_text.strip()), int(period_text.strip())
                    if quota > 0 and period > 0:
                        quotas.append(max(1, quota // period))
        except (TypeError, ValueError):
            # A malformed/partially updated controller file must not make the
            # application unusable. The physical/affinity caps still apply.
            pass

        if current == mount_point:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        try:
            if os.path.commonpath((parent, mount_point)) != mount_point:
                break
        except ValueError:
            break
        current = parent
    return quotas


def _get_cgroup_cpu_capacity() -> int | None:
    """Return the tightest finite CPU quota for this Linux cgroup, if any."""
    cgroup_text = _read_optional_system_text("/proc/self/cgroup")
    mountinfo_text = _read_optional_system_text("/proc/self/mountinfo")
    if cgroup_text is None or mountinfo_text is None:
        return None

    unified_path: str | None = None
    controller_paths: dict[str, str] = {}
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, cgroup_path = fields
        if hierarchy == "0" and not controllers:
            unified_path = cgroup_path
        for controller in controllers.split(","):
            if controller:
                controller_paths[controller] = cgroup_path

    quotas: list[int] = []
    for line in mountinfo_text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        mount_fields, filesystem_fields = left.split(), right.split()
        if len(mount_fields) < 5 or not filesystem_fields:
            continue
        mount_root = _decode_mountinfo_path(mount_fields[3])
        mount_point = _decode_mountinfo_path(mount_fields[4])
        filesystem_type = filesystem_fields[0]
        if filesystem_type == "cgroup2" and unified_path is not None:
            cgroup_path, version = unified_path, 2
        elif (filesystem_type == "cgroup" and "cpu" in controller_paths
              and "cpu" in set(
                  ",".join(filesystem_fields[1:]).split(","))):
            cgroup_path, version = controller_paths["cpu"], 1
        else:
            continue
        directory = _cgroup_directory(cgroup_path, mount_root, mount_point)
        if directory is not None:
            quotas.extend(_finite_cgroup_cpu_quotas(
                directory, mount_point, version))

    return min(quotas) if quotas else None


def get_mdrun_cpu_capacity() -> int:
    """Return a conservative CPU-slot budget for concurrent mdrun processes.

    Physical cores are preferred over SMT/logical threads.  A restricted CPU
    affinity and finite cgroup CPU quota further cap that value.
    """
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True) or os.cpu_count()
    base = physical if isinstance(physical, int) and physical > 0 else logical
    if not isinstance(base, int) or base < 1:
        base = 1

    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except (OSError, TypeError, ValueError):
            affinity_count = None
    if isinstance(affinity_count, int) and affinity_count > 0:
        base = min(base, affinity_count)
    cgroup_capacity = _get_cgroup_cpu_capacity()
    if isinstance(cgroup_capacity, int) and cgroup_capacity > 0:
        base = min(base, cgroup_capacity)
    return max(1, base)


def _describe_process_resources(cpu_slots: int, use_gpu: bool) -> str:
    cpu_label = "CPU slot" if cpu_slots == 1 else "CPU slots"
    description = f"Reserved {cpu_slots} {cpu_label}"
    if use_gpu:
        description += " and exclusive GPU use"
    return description + "."


def reserve_process_resources(job_key: str, mpi_ranks: int,
                              omp_threads: int, use_gpu: bool, *,
                              request_label: str =
                              "MPI Ranks × OpenMP Threads",
                              reduction_hint: str =
                              "reduce MPI Ranks / OpenMP Threads") -> str:
    """Atomically admit one reserved simulation into the host resource budget.

    A job key owns at most one reservation, so attaching another browser session
    to the registered process never consumes the resources a second time.
    """
    for value, label in ((mpi_ranks, "MPI ranks"),
                         (omp_threads, "OpenMP threads")):
        if (not isinstance(value, int) or isinstance(value, bool)
                or value < 1):
            raise ValueError(f"{label} must be a positive integer.")
    if not isinstance(use_gpu, bool):
        raise ValueError("Use GPU must be true or false.")

    requested_cpu = mpi_ranks * omp_threads
    capacity = get_mdrun_cpu_capacity()
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        _prune_finished_process_jobs_unlocked()
        existing = _PROCESS_RESOURCE_RESERVATIONS.get(canonical_job_key)
        if existing is not None:
            # Idempotence is intentional for a refreshed/attached UI session.
            return _describe_process_resources(*existing)

        if (_PROCESS_REGISTRY.get(canonical_job_key) is not _PROCESS_SLOT_RESERVED
                or _PROCESS_RESERVATION_OWNERS.get(canonical_job_key)
                != threading.get_ident()):
            raise RuntimeError(
                "Simulation resources can only be reserved by the callback that "
                "owns the process launch slot.")

        used_cpu = sum(reservation[0]
                       for reservation in _PROCESS_RESOURCE_RESERVATIONS.values())
        available_cpu = max(0, capacity - used_cpu)
        if requested_cpu > available_cpu:
            raise ResourceAdmissionError(
                f"This run requests {requested_cpu} CPU slots ({request_label}), "
                f"but {used_cpu} of the server's conservative "
                f"{capacity}-slot budget are already reserved and only "
                f"{available_cpu} remain. Stop or wait for another compute job, "
                f"or {reduction_hint}.")
        if use_gpu and any(reservation[1]
                           for reservation in
                           _PROCESS_RESOURCE_RESERVATIONS.values()):
            raise ResourceAdmissionError(
                "The server GPU is already reserved by another simulation. Stop "
                "or wait for that run to finish, or clear Use GPU for this run.")

        _PROCESS_RESOURCE_RESERVATIONS[canonical_job_key] = (
            requested_cpu, use_gpu)
        return _describe_process_resources(requested_cpu, use_gpu)


def get_process_resource_summary(job_key: str | None) -> str | None:
    """Describe the existing admission for status shown by an attached session."""
    if job_key is None:
        return None
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        reservation = _PROCESS_RESOURCE_RESERVATIONS.get(canonical_job_key)
    return (_describe_process_resources(*reservation)
            if reservation is not None else None)


def _release_process_resources_unlocked(canonical_job_key: str) -> None:
    """Release one admission while the caller holds the registry lock."""
    _PROCESS_RESOURCE_RESERVATIONS.pop(canonical_job_key, None)


def _registered_process_is_active_unlocked(registered: object) -> bool:
    """Whether a registry entry or any child in its private group is active."""
    if registered is _PROCESS_SLOT_RESERVED:
        return True
    try:
        if registered.poll() is None:  # type: ignore[attr-defined]
            return True
        process_group = _owned_process_group(registered)  # type: ignore[arg-type]
        return (process_group is not None
                and _process_group_has_live_members(process_group))
    except Exception:
        # An unqueryable child may still be writing.  Keep both the output and
        # resource reservations charged on that uncertainty.
        return True


def _prune_finished_process_jobs_unlocked() -> None:
    """Discard completed registry entries before computing global capacity.

    Watchers normally release admissions immediately.  This lazy pass keeps a
    failed/delayed watcher from permanently consuming the whole host budget,
    while retaining an exited MPI leader until every member of its private
    process group has gone.
    """
    for job_key, registered in list(_PROCESS_REGISTRY.items()):
        if _registered_process_is_active_unlocked(registered):
            continue
        _PROCESS_REGISTRY.pop(job_key, None)
        _PROCESS_REGISTRY_METADATA.pop(job_key, None)
        _PROCESS_RESERVATION_OWNERS.pop(job_key, None)
        _release_process_resources_unlocked(job_key)

    # Defensive consistency repair for an interrupted/custom integration that
    # removed a registry entry without going through release_process_job().
    for job_key in list(_PROCESS_RESOURCE_RESERVATIONS):
        if job_key not in _PROCESS_REGISTRY:
            _release_process_resources_unlocked(job_key)


def _registered_writer_in_directory_unlocked(
        canonical_directory: str) -> tuple[str, object] | None:
    """Find a live/reserved registry entry, pruning completed children.

    The caller must hold ``_PROCESS_REGISTRY_LOCK``.
    """
    for job_key, registered in list(_PROCESS_REGISTRY.items()):
        job_directory = os.path.dirname(job_key)
        if not _directories_overlap(job_directory, canonical_directory):
            continue
        if _registered_process_is_active_unlocked(registered):
            return job_key, registered
        del _PROCESS_REGISTRY[job_key]
        _PROCESS_REGISTRY_METADATA.pop(job_key, None)
        _PROCESS_RESERVATION_OWNERS.pop(job_key, None)
        _release_process_resources_unlocked(job_key)
    return None


@contextmanager
def reserve_working_directory_maintenance(
        working_directory_path: str) -> Iterator[None]:
    """Exclusively reserve a job directory for a short file mutation.

    Process starts and other mutations are rejected until the context exits.
    Conversely, a live process or an in-progress process launch prevents the
    mutation.  Acquisition is atomic with ``reserve_process_job``.
    """
    canonical_directory = _canonical_process_path(working_directory_path)
    owner_thread = threading.get_ident()
    with _PROCESS_REGISTRY_LOCK:
        if any(_directories_overlap(canonical_directory, reserved_directory)
               and reservation_owner != owner_thread
               for reserved_directory, (reservation_owner, _)
               in _DIRECTORY_MAINTENANCE_RESERVATIONS.items()):
            raise WorkingDirectoryBusyError(
                "Another file operation is already using this working directory."
            )
        if any(_directories_overlap(canonical_directory, reserved_directory)
               and read_owner != owner_thread
               for reserved_directory, read_owner
               in _DIRECTORY_READ_RESERVATIONS.items()):
            raise WorkingDirectoryBusyError(
                "An analysis or viewer is using this working directory."
            )
        registered_writer = _registered_writer_in_directory_unlocked(
            canonical_directory)
        if registered_writer is not None:
            writer_key, writer = registered_writer
            if not (writer is _PROCESS_SLOT_RESERVED
                    and _PROCESS_RESERVATION_OWNERS.get(writer_key)
                    == owner_thread):
                raise WorkingDirectoryBusyError(
                    "A simulation or analysis job is running in this working "
                    "directory. Stop it or wait for it to finish before changing "
                    "files."
                )

        reservation = _DIRECTORY_MAINTENANCE_RESERVATIONS.get(
            canonical_directory)
        depth = reservation[1] + 1 if reservation is not None else 1
        _DIRECTORY_MAINTENANCE_RESERVATIONS[canonical_directory] = (
            owner_thread, depth)

    try:
        yield
    finally:
        with _PROCESS_REGISTRY_LOCK:
            reservation = _DIRECTORY_MAINTENANCE_RESERVATIONS.get(
                canonical_directory)
            if (reservation is not None and reservation[0] == owner_thread
                    and reservation[1] > 1):
                _DIRECTORY_MAINTENANCE_RESERVATIONS[canonical_directory] = (
                    owner_thread, reservation[1] - 1)
            elif reservation is not None and reservation[0] == owner_thread:
                _DIRECTORY_MAINTENANCE_RESERVATIONS.pop(
                    canonical_directory, None)


@contextmanager
def reserve_working_directory_read(
        working_directory_path: str) -> Iterator[None]:
    """Exclusively snapshot a job's logical state for a read/analysis callback."""
    canonical_directory = _canonical_process_path(working_directory_path)
    owner_thread = threading.get_ident()
    with _PROCESS_REGISTRY_LOCK:
        if any(_directories_overlap(canonical_directory, reserved_directory)
               for reserved_directory in _DIRECTORY_MAINTENANCE_RESERVATIONS):
            raise WorkingDirectoryBusyError(
                "A file operation is using this working directory."
            )
        if any(_directories_overlap(canonical_directory, reserved_directory)
               for reserved_directory in _DIRECTORY_READ_RESERVATIONS):
            raise WorkingDirectoryBusyError(
                "Another analysis or viewer is using this working directory."
            )
        if _registered_writer_in_directory_unlocked(canonical_directory) is not None:
            raise WorkingDirectoryBusyError(
                "A simulation or analysis job is still running in this working "
                "directory. Stop it or wait for it to finish before reading its results."
            )
        _DIRECTORY_READ_RESERVATIONS[canonical_directory] = owner_thread

    try:
        yield
    finally:
        with _PROCESS_REGISTRY_LOCK:
            _DIRECTORY_READ_RESERVATIONS.pop(canonical_directory, None)


def guard_working_directory_read(callback: Any) -> Any:
    """Hold an exclusive job lease while a callback reads mutable artifacts.

    Long-running simulations and analysis commands can otherwise append to a
    selected trajectory/result while another browser session is parsing it.
    The lease also prevents file-manager mutations.  Commands launched by the
    guarded callback itself are allowed because ownership is thread-scoped.
    """
    signature = inspect.signature(callback)
    if "working_directory_path" not in signature.parameters:
        raise ValueError("Guarded callbacks must accept working_directory_path.")

    def directory_from(args: Any, kwargs: Any) -> str:
        bound = signature.bind(*args, **kwargs)
        return str(bound.arguments["working_directory_path"])

    if inspect.isgeneratorfunction(callback):
        @wraps(callback)
        def guarded_generator(*args: Any, **kwargs: Any) -> Any:
            directory = directory_from(args, kwargs)
            with reserve_working_directory_read(directory):
                canonical_directory = _canonical_process_path(directory)
                iterator = iter(callback(*args, **kwargs))
                while True:
                    # Gradio may advance one synchronous generator on different
                    # anyio workers. Transfer the lease immediately before each
                    # step so that this callback's own managed command remains
                    # authorized without opening it to unrelated threads.
                    with _PROCESS_REGISTRY_LOCK:
                        if canonical_directory in _DIRECTORY_READ_RESERVATIONS:
                            _DIRECTORY_READ_RESERVATIONS[canonical_directory] = \
                                threading.get_ident()
                    try:
                        result = next(iterator)
                    except StopIteration:
                        return
                    yield result
        return guarded_generator

    @wraps(callback)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        directory = directory_from(args, kwargs)
        with reserve_working_directory_read(directory):
            return callback(*args, **kwargs)
    return guarded


def guard_working_directory_reads(namespace: dict[str, Any],
                                  callback_names: Sequence[str]) -> None:
    """Apply :func:`guard_working_directory_read` to named module callbacks."""
    for callback_name in callback_names:
        callback = namespace.get(callback_name)
        if callable(callback):
            namespace[callback_name] = guard_working_directory_read(callback)


def get_process_job_key(working_directory_path: str, output_prefix: str) -> str:
    """Return the canonical registry key for a long-running output target."""
    return os.path.normcase(os.path.realpath(os.path.join(working_directory_path,
                                                          output_prefix)))


def reserve_process_job(job_key: str) -> tuple[bool, subprocess.Popen[str] | None]:
    """Atomically reserve an output target, returning any process using it.

    The reservation closes the otherwise unavoidable gap between checking the
    registry and calling ``Popen``.  Finished entries are discarded lazily so a
    watcher failure cannot leave an output target blocked forever.
    """
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        current_thread = threading.get_ident()
        if any(_path_is_within_directory(canonical_job_key, directory)
               and owner_thread != current_thread
               for directory, (owner_thread, _)
               in _DIRECTORY_MAINTENANCE_RESERVATIONS.items()):
            return False, None
        if any(_path_is_within_directory(canonical_job_key, directory)
               and owner_thread != current_thread
               for directory, owner_thread in _DIRECTORY_READ_RESERVATIONS.items()):
            return False, None
        registered = _PROCESS_REGISTRY.get(canonical_job_key)
        if registered is _PROCESS_SLOT_RESERVED:
            return False, None
        if registered is not None:
            if _registered_process_is_active_unlocked(registered):
                return False, registered  # type: ignore[return-value]
            del _PROCESS_REGISTRY[canonical_job_key]
            _PROCESS_REGISTRY_METADATA.pop(canonical_job_key, None)
            _PROCESS_RESERVATION_OWNERS.pop(canonical_job_key, None)
            _release_process_resources_unlocked(canonical_job_key)

        # Serialize every writer within a job, not merely writers that happen
        # to use the same output prefix.  A same-thread reserved slot is the
        # synchronous caller's outer transaction (for example minimisation)
        # and may launch its managed helper child before publishing results.
        job_directory = os.path.dirname(canonical_job_key)
        conflict = _registered_writer_in_directory_unlocked(job_directory)
        if conflict is not None:
            conflict_key, conflict_process = conflict
            if not (conflict_process is _PROCESS_SLOT_RESERVED
                    and _PROCESS_RESERVATION_OWNERS.get(conflict_key)
                    == current_thread):
                return False, None

        _PROCESS_REGISTRY[canonical_job_key] = _PROCESS_SLOT_RESERVED
        _PROCESS_RESERVATION_OWNERS[canonical_job_key] = current_thread
        return True, None


def register_process_job(job_key: str, proc: subprocess.Popen[str],
                         job_name: str | None = None,
                         working_directory_path: str | None = None,
                         failure_hint: str | None = None) -> None:
    """Attach the newly spawned process to a previously reserved output slot."""
    capture_owned_process_group(proc)
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        if _PROCESS_REGISTRY.get(canonical_job_key) is not _PROCESS_SLOT_RESERVED:
            raise RuntimeError("The process output slot is no longer reserved.")
        _PROCESS_REGISTRY[canonical_job_key] = proc
        _PROCESS_RESERVATION_OWNERS.pop(canonical_job_key, None)
        if job_name is not None and working_directory_path is not None:
            canonical_directory = os.path.realpath(working_directory_path)
            _PROCESS_REGISTRY_METADATA[canonical_job_key] = (
                job_name, canonical_directory, failure_hint)
            # Metadata on the process survives the brief race where it exits and
            # its registry entry is released just as a refreshed session adopts
            # it. Popen instances are ordinary Python objects and support these
            # private attributes.
            try:
                setattr(proc, "_gromacs_webui_job_metadata",
                        (job_name, canonical_directory, failure_hint))
            except Exception:
                pass


def get_registered_process_metadata(job_key: str,
                                    proc: subprocess.Popen[str]) -> tuple[
                                        str, str, str | None] | None:
    """Describe a matching live registry entry for a newly attached session."""
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        if _PROCESS_REGISTRY.get(canonical_job_key) is proc:
            metadata = _PROCESS_REGISTRY_METADATA.get(canonical_job_key)
            if metadata is not None:
                return metadata
    metadata = getattr(proc, "_gromacs_webui_job_metadata", None)
    if (isinstance(metadata, tuple) and len(metadata) == 3
            and isinstance(metadata[0], str) and isinstance(metadata[1], str)):
        return metadata
    return None


def release_process_job(job_key: str | None,
                        proc: subprocess.Popen[str] | None = None) -> None:
    """Release a reservation/process, but never remove a replacement process."""
    if job_key is None:
        return
    canonical_job_key = _canonical_process_path(job_key)
    with _PROCESS_REGISTRY_LOCK:
        registered = _PROCESS_REGISTRY.get(canonical_job_key)
        if ((proc is None and registered is _PROCESS_SLOT_RESERVED)
                or (proc is not None and registered is proc)):
            del _PROCESS_REGISTRY[canonical_job_key]
            _PROCESS_REGISTRY_METADATA.pop(canonical_job_key, None)
            _PROCESS_RESERVATION_OWNERS.pop(canonical_job_key, None)
            _release_process_resources_unlocked(canonical_job_key)


def is_working_directory_busy(working_directory_path: str) -> bool:
    """Whether a registered writer targets a file below this job directory."""
    canonical_directory = _canonical_process_path(working_directory_path)
    with _PROCESS_REGISTRY_LOCK:
        if any(_directories_overlap(canonical_directory, reserved_directory)
               for reserved_directory in _DIRECTORY_MAINTENANCE_RESERVATIONS):
            return True
        if any(_directories_overlap(canonical_directory, reserved_directory)
               for reserved_directory in _DIRECTORY_READ_RESERVATIONS):
            return True
        return _registered_writer_in_directory_unlocked(canonical_directory) is not None


def stop_all_registered_processes(timeout: float = 15) -> int:
    """Stop and reap every distinct live child registered by this server.

    Registry ownership is cleared atomically before any potentially blocking
    process operation. This is intended for server shutdown, when no new launch
    should occur; callers must arrange that lifecycle ordering.
    """
    if isinstance(timeout, bool):
        raise ValueError("Shutdown timeout must be finite and non-negative.")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        raise ValueError("Shutdown timeout must be finite and non-negative.") from None
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Shutdown timeout must be finite and non-negative.")
    deadline = time.monotonic() + timeout

    with _PROCESS_REGISTRY_LOCK:
        registered_processes = list(_PROCESS_REGISTRY.values())
        _PROCESS_REGISTRY.clear()
        _PROCESS_REGISTRY_METADATA.clear()
        _PROCESS_RESERVATION_OWNERS.clear()
        _PROCESS_RESOURCE_RESERVATIONS.clear()

    stopped = 0
    seen: set[int] = set()
    stop_errors: list[Exception] = []
    for proc in registered_processes:
        if proc is _PROCESS_SLOT_RESERVED or id(proc) in seen:
            continue
        seen.add(id(proc))
        try:
            running = proc.poll() is None  # type: ignore[attr-defined]
            process_group = _owned_process_group(proc)  # type: ignore[arg-type]
            running = running or (
                process_group is not None
                and _process_group_has_live_members(process_group))
        except Exception:
            # Only Popen objects may be registered; remain defensive in case a
            # malformed test/integration mutates the private registry.
            continue
        if not running:
            continue
        try:
            setattr(proc, "_gromacs_webui_stopped_by_user", True)
        except Exception:
            pass
        try:
            # ``timeout`` is a budget for the whole server shutdown, not a fresh
            # delay for every job.  Later jobs are killed immediately once an
            # earlier stubborn process has consumed the grace period.
            remaining = max(0.0, deadline - time.monotonic())
            stop_process_gracefully(proc, timeout=remaining)  # type: ignore[arg-type]
            stopped += 1
        except Exception as exc:
            # Shutdown must still attempt every independent child even if one
            # unusual Popen/OS error prevents a clean stop.
            stop_errors.append(exc)
    if stop_errors:
        raise RuntimeError(
            f"Failed to stop {len(stop_errors)} registered process(es); "
            f"first error: {stop_errors[0]}"
        ) from stop_errors[0]
    return stopped


def set_process_running(process_state: ProcessStateDict, proc: subprocess.Popen[str],
                        job_key: str, job_name: str, working_directory_path: str,
                        failure_hint: str | None = None) -> None:
    """Associate a UI state with a registered server process."""
    with process_state["lock"]:
        process_state.update({
            "proc": proc,
            "running": True,
            "job_key": job_key,
            "job_name": job_name,
            "working_directory": os.path.realpath(working_directory_path),
            "returncode": None,
            "completion_status": None,
            "completion_color": None,
            "completion_pending": False,
            "failure_hint": failure_hint,
        })


def _reset_process_state_unlocked(process_state: ProcessStateDict) -> None:
    """Reset state while its caller holds ``process_state['lock']``."""
    process_state.update({
        "proc": None,
        "running": False,
        "job_key": None,
        "job_name": None,
        "working_directory": None,
        "returncode": None,
        "completion_status": None,
        "completion_color": None,
        "completion_pending": False,
        "failure_hint": None,
    })


def clear_process_state(process_state: ProcessStateDict) -> tuple[
        subprocess.Popen[str] | None, str | None]:
    """Detach and return the current process and key before stopping it."""
    with process_state["lock"]:
        proc = process_state.get("proc") if process_state.get("running") else None
        job_key = process_state.get("job_key") if proc is not None else None
        _reset_process_state_unlocked(process_state)
    return proc, job_key


def clear_process_state_for_directory(process_state: ProcessStateDict,
                                      working_directory_path: str) -> tuple[
                                          subprocess.Popen[str] | None, str | None]:
    """Take this directory's process for Stop, or detach an older directory.

    A user may switch jobs while an old simulation keeps running.  Pressing Run
    in the new job must start that job, not unexpectedly stop the old one.  The
    old child remains protected by the global registry and its watcher still
    releases the slot when it exits.
    """
    current_directory = os.path.realpath(working_directory_path)
    with process_state["lock"]:
        associated_directory = process_state.get("working_directory")
        if (process_state.get("running") and associated_directory
                and associated_directory != current_directory):
            _reset_process_state_unlocked(process_state)
            return None, None

        proc = process_state.get("proc") if process_state.get("running") else None
        job_key = process_state.get("job_key") if proc is not None else None
        _reset_process_state_unlocked(process_state)
    return proc, job_key


def clear_process_state_if_current(process_state: ProcessStateDict,
                                   expected_proc: subprocess.Popen[str] | None) -> bool:
    """Clear launch state only if no newer process has replaced it."""
    with process_state["lock"]:
        if process_state.get("proc") is not expected_proc:
            return False
        _reset_process_state_unlocked(process_state)
    return True


def watch_process(proc: subprocess.Popen[str], process_state: ProcessStateDict,
                  job_key: str | None = None) -> None:
    """Record completion without letting an old watcher clear a newer run.

    Checking object identity is essential here.  Merely checking ``running``
    has an ABA race: run A can be stopped, run B can start, and A's watcher can
    then wake up and incorrectly mark B as finished.
    """
    if job_key is None:
        with process_state["lock"]:
            if process_state.get("proc") is proc:
                job_key = process_state.get("job_key")

    wait_error: Exception | None = None
    try:
        returncode = proc.wait()
    except Exception as exc:  # pragma: no cover - defensive around Popen.wait
        wait_error = exc
        try:
            returncode = proc.poll()
        except Exception:
            returncode = None

    # A failed wait does not prove the child stopped.  Retain both the state and
    # registry slot while it is still live, so an unusual OS/Popen failure cannot
    # open the door to a duplicate writer.  The UI timer will continue polling.
    if wait_error is not None and returncode is None:
        with process_state["lock"]:
            if process_state.get("proc") is proc:
                process_state.update({
                    "completion_status": (
                        f"{process_state.get('job_name') or 'Process'} status watcher "
                        f"failed while the process is still running: {wait_error}"),
                    "completion_color": "red",
                    "completion_pending": True,
                })
        return

    # The registered leader may have spawned MPI ranks/helpers and exited
    # before them.  Retain the slot until its private group has no live members;
    # otherwise a second job can start while an orphan is still writing.
    try:
        stop_process_gracefully(
            proc, timeout=1.0, mark_stopped_by_user=False)
    except Exception as exc:
        if wait_error is None:
            wait_error = exc

    with process_state["lock"]:
        if process_state.get("proc") is proc:
            job_name = process_state.get("job_name") or "Process"
            working_directory = process_state.get("working_directory")
            location = (f" (working directory '{os.path.basename(working_directory)}')"
                        if working_directory else "")
            failure_hint = process_state.get("failure_hint") or ""
            if wait_error is not None:
                message = f"{job_name} status watcher failed{location}: {wait_error}"
                color = "red"
            elif getattr(proc, "_gromacs_webui_stopped_by_user", False):
                message = f"{job_name} stopped by user{location}."
                color = "red"
            elif returncode == 0:
                message = f"{job_name} completed successfully{location}."
                color = "green"
            else:
                message = f"{job_name} failed with exit code {returncode}{location}."
                if failure_hint:
                    message += f" {failure_hint}"
                color = "red"

            process_state.update({
                "proc": None,
                "running": False,
                "returncode": returncode,
                "completion_status": message,
                "completion_color": color,
                "completion_pending": True,
            })

    # Matching in the registry independently protects a replacement process,
    # including one associated with a different browser session.
    release_process_job(job_key, proc)


def refresh_process_state(process_state: ProcessStateDict) -> None:
    """Notice completion even if a watcher thread could not update this state."""
    with process_state["lock"]:
        proc = process_state.get("proc") if process_state.get("running") else None
    if proc is None:
        return
    try:
        returncode = proc.poll()
    except Exception:
        return
    if returncode is None:
        return

    # ``watch_process`` calls wait(), but a process which poll() has reaped can
    # safely be waited on again.  Passing a tiny adapter would weaken the same-
    # object identity guard, so let the common watcher path do the bookkeeping.
    watch_process(proc, process_state)


def consume_process_completion(process_state: ProcessStateDict) -> tuple[
        bool, str | None, str | None, str | None]:
    """Return running/status/directory and consume a one-shot completion notice."""
    with process_state["lock"]:
        running = bool(process_state.get("running"))
        if not process_state.get("completion_pending"):
            return running, None, None, None
        message = process_state.get("completion_status")
        color = process_state.get("completion_color")
        directory = process_state.get("working_directory")
        process_state["completion_pending"] = False
    return running, message, color, directory


def get_process_job_name(process_state: ProcessStateDict,
                         default: str = "A long-running job") -> str:
    """Read the associated job label without bypassing the state's lock."""
    with process_state["lock"]:
        return process_state.get("job_name") or default

DEFAULT_TERMINUS_CHOICE: str = "Default (charged)"

# Terminus patch names offered in the UI. The list a given force field actually
# accepts comes from its aminoacids.[nc].tdb, and pdb2gmx filters it per residue,
# so the dropdowns allow custom values and the real menu is resolved at run time
# by resolve_terminus_selections().
N_TERMINUS_CHOICES: list[str] = [DEFAULT_TERMINUS_CHOICE, "NH3+", "NH2", "None", "GLY-NH3+", "PRO-NH2+"]
C_TERMINUS_CHOICES: list[str] = [DEFAULT_TERMINUS_CHOICE, "COO-", "COOH", "CT1", "CT2", "None"]

PROBE_PDB2GMX_PREFIX: str = ".probe_pdb2gmx"

def _parse_terminus_menus(pdb2gmx_output: str) -> list[TerminusMenu]:
    """Extract each terminus menu, in the order pdb2gmx asked about them."""
    # choose_ter() prints the title with printf(), then one "%2d: name" line per
    # option, so the options are the lines directly following a title line.
    menus = []
    current_menu = None

    for line in pdb2gmx_output.splitlines():
        title = re.match(r'\s*Select (start|end) terminus type for (\S+)', line)
        if title:
            current_menu = {"kind": title.group(1), "residue": title.group(2), "options": []}
            menus.append(current_menu)
            continue

        if current_menu is None:
            continue

        option = re.match(r'\s*(\d+):\s*(.+?)\s*$', line)
        if option:
            # choose_ter() appends a note to zwitterion entries; keep the name only.
            label = re.sub(r'\s*\(only use with zwitterions.*\)\s*$', '', option.group(2)).strip()
            current_menu["options"].append((option.group(1), label))
        else:
            current_menu = None

    return menus

def _remove_pdb2gmx_probe_files(working_directory_path: str,
                                probe_prefix: str = PROBE_PDB2GMX_PREFIX) -> None:
    """Delete one probe's throwaway outputs and GROMACS backup files."""
    try:
        names = os.listdir(working_directory_path)
    except OSError:
        return

    for name in names:
        # pdb2gmx also writes per-chain include files and #backup# copies derived
        # from the probe output names.
        if name.startswith(probe_prefix) or name.startswith("#" + probe_prefix):
            try:
                os.remove(os.path.join(working_directory_path, name))
            except OSError:
                pass

def resolve_terminus_selections(pdb2gmx_cmd: Sequence[str], working_directory_path: str,
                                n_terminus: str | None,
                                c_terminus: str | None) -> tuple[str | None, list[str]]:
    """Probe 'gmx pdb2gmx -ter' to read the terminus menu it prints for each chain,
    then map the requested terminus names onto the stdin answers for the real run.

    Returns (stdin_input, descriptions), or (None, []) when the force field offers
    no terminus choice. Matching is done per prompt rather than with a fixed index
    because filter_ter() builds a different list per residue (a PRO N-terminus
    offers PRO-NH2+ first, so NH3+ is not index 0 there)."""
    # The real command runs with cwd set to the working directory and plain file
    # names, so the probe uses plain names too.
    probe_prefix = f"{PROBE_PDB2GMX_PREFIX}_{uuid.uuid4().hex}"
    probe_cmd = list(pdb2gmx_cmd)
    probe_cmd[probe_cmd.index("-o") + 1] = probe_prefix + ".gro"
    probe_cmd[probe_cmd.index("-p") + 1] = probe_prefix + ".top"
    probe_itp = probe_prefix + ".itp"
    if "-i" in probe_cmd:
        probe_cmd[probe_cmd.index("-i") + 1] = probe_itp
    else:
        probe_cmd.extend(["-i", probe_itp])

    try:
        # Answer every prompt up front: pdb2gmx's menu goes to a block-buffered
        # stream, so reading it before replying would deadlock.
        probe = run_managed_command(
            probe_cmd, cwd=working_directory_path, stdin_input="0\n" * 512)
        stdout_probe, stderr_probe = probe.stdout, probe.stderr
    finally:
        _remove_pdb2gmx_probe_files(working_directory_path, probe_prefix)

    if probe.returncode != 0:
        raise Exception(stderr_probe)

    menus = _parse_terminus_menus(stdout_probe)
    if not menus:
        # Some force fields (the AMBER ports, for instance) apply terminus patches
        # through renamed terminal residues and offer no choice at all. Report that
        # rather than failing, so the caller can fall back to the default run.
        return None, []

    answers = []
    descriptions = []
    for menu in menus:
        requested = n_terminus if menu["kind"] == "start" else c_terminus
        if requested is None or requested.strip() == "" or requested == DEFAULT_TERMINUS_CHOICE:
            index, label = menu["options"][0]
        else:
            match = next((option for option in menu["options"] if option[1].lower() == requested.strip().lower()), None)
            if match is None:
                available = ", ".join(label for _, label in menu["options"])
                raise Exception(f"Terminus type '{requested}' is not available for the {menu['kind']} terminus of {menu['residue']}.\nAvailable types: {available}")
            index, label = match

        answers.append(index)
        descriptions.append(f"{menu['residue']} {label}")

    return "\n".join(answers) + "\n", descriptions

def get_force_field_family(force_field: str | None) -> str | None:
    """Return the recognised force-field family that controls MDP cutoffs."""
    normalised = str(force_field or "").strip().lower()
    for prefix, family in (
        ("amber", "AMBER"),
        ("charmm", "CHARMM"),
        ("gromos", "GROMOS"),
        ("opls", "OPLS"),
    ):
        if normalised.startswith(prefix):
            return family
    return None


def is_charmm_force_field(force_field: str | None) -> bool:
    """Report whether a force field name belongs to the CHARMM family."""
    return get_force_field_family(force_field) == "CHARMM"


def is_gromos_force_field(force_field: str | None) -> bool:
    """Report whether a force field name belongs to the GROMOS family."""
    return get_force_field_family(force_field) == "GROMOS"


def uses_long_range_dispersion_correction(force_field: str | None) -> bool:
    """Whether this force-field family uses an LJ energy/pressure tail correction."""
    return get_force_field_family(force_field) in ("AMBER", "OPLS")


def get_cutoff_mdp_section(force_field: str | None) -> str:
    """Return the neighbour-searching and cutoff block suited to the force field."""
    if is_charmm_force_field(force_field):
        # CHARMM is parameterised with a force-switched LJ potential; plain
        # cutoffs at 1.0 nm give wrong energetics.
        return """; Neighbor searching and cutoffs (CHARMM force-switch)
cutoff-scheme   = Verlet
vdwtype         = cutoff
vdw-modifier    = force-switch
rlist           = 1.2
rvdw            = 1.2
rvdw-switch     = 1.0
coulombtype     = PME
rcoulomb        = 1.2
DispCorr        = no"""

    if is_gromos_force_field(force_field):
        # GROMOS96 was parameterised with a 1.4 nm long-range cutoff. A shorter
        # Verlet cutoff changes the force field rather than merely saving work.
        return """; Neighbor searching and cutoffs (GROMOS long-range cutoff)
cutoff-scheme   = Verlet
rlist           = 1.4
rvdw            = 1.4
rcoulomb        = 1.4
coulombtype     = PME
DispCorr        = no"""

    dispersion_correction = (
        "\nDispCorr        = EnerPres"
        if uses_long_range_dispersion_correction(force_field)
        else ""
    )
    return f"""; Neighbor searching and cutoffs
cutoff-scheme   = Verlet
rlist           = 1.0
rvdw            = 1.0
rcoulomb        = 1.0
coulombtype     = PME{dispersion_correction}"""

def get_default_ion_addition_mdp_file_content(force_field: str | None = None) -> str:
    """MDP for the short minimisation that precedes ion placement."""
    return f"""
integrator = steep
nsteps     = 500
emtol      = 1000.0

{get_cutoff_mdp_section(force_field)}
"""

def get_default_energy_minimization_mdp_file_content(force_field: str | None = None) -> str:
    """MDP for steepest-descent energy minimisation."""
    return f"""
integrator  = steep
nsteps      = 50000
emtol       = 1000.0

{get_cutoff_mdp_section(force_field)}
"""


def _positive_finite_number(value: Any, label: str) -> float:
    """Return a finite positive float for a generated simulation setting."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and greater than zero.")
    return number


def _exact_integer_in_range(value: Any, label: str,
                            minimum: int, maximum: int) -> int:
    """Parse an integer-valued input without silently truncating fractions."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{label} must be an integer from {minimum} to {maximum}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be an integer from {minimum} to {maximum}.") from None
    if (not math.isfinite(numeric) or not numeric.is_integer()
            or numeric < minimum or numeric > maximum):
        raise ValueError(
            f"{label} must be an integer from {minimum} to {maximum}.")
    return int(numeric)


GROMACS_ION_NAME_CHARGES: dict[str, frozenset[int]] = {
    # Conservative names shared by the commonly bundled GROMACS force fields.
    "NA": frozenset({1}),
    "K": frozenset({1}),
    "LI": frozenset({1}),
    "MG": frozenset({2}),
    "CA": frozenset({2}),
    "ZN": frozenset({2}),
    "CL": frozenset({-1}),
    "F": frozenset({-1}),
    "BR": frozenset({-1}),
    "I": frozenset({-1}),
}


_ION_RESIDUE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{0,4}")


def validate_ion_species_charges(cation_name: Any, cation_charge: int,
                                 anion_name: Any,
                                 anion_charge: int) -> tuple[str, str]:
    """Validate ion residue names and reject known names with a wrong valence.

    The mapping is deliberately conservative. Unlisted names remain available
    for custom force fields, whose ion residue names cannot be inferred here.
    """
    normalized_names: list[str] = []
    for role, name, charge in (
            ("Cation", cation_name, cation_charge),
            ("Anion", anion_name, anion_charge)):
        if not isinstance(name, str) or not _ION_RESIDUE_NAME_RE.fullmatch(name):
            raise ValueError(
                f"{role} ion residue name must contain 1 to 5 letters, digits, "
                "underscore, plus, or minus characters and start with a letter."
            )
        # GROMACS topology identifiers are case-sensitive.  Canonicalise only
        # the built-in aliases that we know about; preserving the spelling of
        # an unlisted name is essential for custom force fields (for example,
        # a molecule type named ``Cat`` is not interchangeable with ``CAT``).
        lookup_name = name.upper()
        allowed_charges = GROMACS_ION_NAME_CHARGES.get(lookup_name)
        normalized_names.append(lookup_name if allowed_charges is not None else name)
        if allowed_charges is None or charge in allowed_charges:
            continue
        allowed = " or ".join(f"{value:+d}" for value in sorted(allowed_charges))
        raise ValueError(
            f"{role} ion residue name '{name}' requires charge {allowed}; "
            f"received {charge:+d}.")
    return normalized_names[0], normalized_names[1]


def validate_ionized_system_with_grompp(
        structure_file_path: str, topology_file_path: str,
        working_directory_path: str, runner: Any = None) -> None:
    """Prove that genion's residue names exist in the selected topology.

    ``gmx genion`` accepts arbitrary names and charges and can exit successfully
    even when the force field has no corresponding ``[ moleculetype ]``.  The
    failure then appears only at the next simulation step.  A disposable grompp
    run catches that contract violation before the staged GRO/topology pair is
    published.
    """
    force_field = get_topology_force_field_name(topology_file_path)
    if force_field is None:
        raise ValueError(
            "The ionized topology has no <force-field>.ff/forcefield.itp "
            "include, so its ion species cannot be validated."
        )
    if runner is None:
        runner = run_checked_command

    stage_directory = os.path.dirname(os.path.realpath(topology_file_path))
    validation_mdp = os.path.join(stage_directory, ".validate_ions.mdp")
    validation_tpr = os.path.join(stage_directory, ".validate_ions.tpr")
    validation_processed_mdp = os.path.join(
        stage_directory, ".validate_ions_processed.mdp")
    with open(validation_mdp, "w", encoding="utf-8") as handle:
        handle.write(get_default_ion_addition_mdp_file_content(force_field))

    command = [
        "gmx", "grompp",
        "-f", validation_mdp,
        "-c", structure_file_path,
        "-p", topology_file_path,
        "-o", validation_tpr,
        "-po", validation_processed_mdp,
        "-maxwarn", "0",
    ]
    run_grompp_with_gromos_warning_policy(
        command, working_directory_path, topology_file_path, 0,
        runner=runner)


def validate_ion_addition_parameters(
        mode: Any, concentration_millimolar: Any,
        cation_charge: Any, anion_charge: Any,
        number_of_cations: Any, number_of_anions: Any,
        neutralize: Any,
) -> tuple[str, float | None, int, int, int | None, int | None, bool]:
    """Validate charges plus only the numeric fields used by the chosen mode."""
    if mode not in ("Concentration", "Number"):
        raise ValueError(
            "Ion addition mode must be either Concentration or Number.")
    if type(neutralize) is not bool:
        raise ValueError("Neutralize must be a boolean value.")

    positive_charge = _exact_integer_in_range(
        cation_charge, "Cation charge", 1, 3)
    negative_charge = _exact_integer_in_range(
        anion_charge, "Anion charge", -3, -1)

    concentration: float | None = None
    cation_count: int | None = None
    anion_count: int | None = None
    if mode == "Concentration":
        if isinstance(concentration_millimolar, (bool, np.bool_)):
            raise ValueError(
                "Ion concentration must be a finite number from 0 to 1000 mM.")
        try:
            concentration = float(concentration_millimolar)
        except (TypeError, ValueError):
            raise ValueError(
                "Ion concentration must be a finite number from 0 to 1000 mM.") from None
        if not math.isfinite(concentration) or not 0 <= concentration <= 1000:
            raise ValueError(
                "Ion concentration must be a finite number from 0 to 1000 mM.")
    else:
        cation_count = _exact_integer_in_range(
            number_of_cations, "Number of cations", 0, 100)
        anion_count = _exact_integer_in_range(
            number_of_anions, "Number of anions", 0, 100)
    return (mode, concentration, positive_charge, negative_charge,
            cation_count, anion_count, neutralize)


def _simulation_step_count(time_scale_ps: Any, time_step_ps: Any) -> tuple[float, float, int]:
    """Validate a duration/timestep and return their positive step count.

    GROMACS assigns special meaning to negative ``nsteps`` (run indefinitely),
    so relying only on a browser slider's minimum is unsafe for API calls.
    """
    duration = _positive_finite_number(time_scale_ps, "Simulation duration")
    timestep = _positive_finite_number(time_step_ps, "Time step")
    steps = int(round(duration / timestep))
    if steps < 1:
        raise ValueError("Simulation duration must contain at least one time step.")
    return duration, timestep, steps


def _single_line_configuration_value(value: Any, label: str) -> str:
    """Validate a free-form value before interpolating it into an input file."""
    if value is None:
        raise ValueError(f"{label} must be a non-empty single-line value.")
    text = str(value)
    if (not text.strip() or text != text.strip() or ";" in text
            or any(ord(character) < 32 or ord(character) == 127
                   for character in text)):
        raise ValueError(f"{label} must be a non-empty single-line value.")
    return text


def get_dynamics_constraint_type(force_field: str | None) -> str:
    """Return the bond-constraint policy required at the workflow's 2 fs step."""
    return "all-bonds" if get_force_field_family(force_field) == "GROMOS" else "h-bonds"

def get_default_nvt_equilibration_mdp_file_content(time_scale_ps: float = 500, time_step_ps: float = 0.002,
                                                   temperature: float = 300, with_ligand: bool = False,
                                                   force_field: str | None = None) -> str:
    """MDP for restrained NVT equilibration with freshly generated velocities."""
    _, time_step_ps, nsteps = _simulation_step_count(time_scale_ps, time_step_ps)
    temperature = _positive_finite_number(temperature, "Temperature")
    restraint_defines = "-DPOSRES -DPOSRES_LIG" if with_ligand else "-DPOSRES"
    constraint_type = get_dynamics_constraint_type(force_field)
    return f"""
; Restrain the solute while the solvent relaxes around it
define      = {restraint_defines}

integrator  = md
dt          = {time_step_ps}
nsteps      = {nsteps}
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
constraints = {constraint_type}

; Energy minimisation leaves no velocities behind, so draw them from a
; Maxwell distribution instead of starting the thermostat from 0 K
continuation = no
gen_vel     = yes
gen_temp    = {temperature}
gen_seed    = -1

{get_cutoff_mdp_section(force_field)}
"""

def get_default_npt_equilibration_mdp_file_content(time_scale_ps: float = 1000, time_step_ps: float = 0.002,
                                                   temperature: float = 300, pressure: float = 1.0,
                                                   with_ligand: bool = False,
                                                   force_field: str | None = None) -> str:
    """MDP for restrained NPT equilibration under the C-rescale barostat."""
    _, time_step_ps, nsteps = _simulation_step_count(time_scale_ps, time_step_ps)
    temperature = _positive_finite_number(temperature, "Temperature")
    pressure = _positive_finite_number(pressure, "Pressure")
    restraint_defines = "-DPOSRES -DPOSRES_LIG" if with_ligand else "-DPOSRES"
    constraint_type = get_dynamics_constraint_type(force_field)
    return f"""
; Keep the solute restrained through the density equilibration as well
define          = {restraint_defines}

integrator      = md
dt              = {time_step_ps}
nsteps          = {nsteps}

; Output control
nstxout         = 1000
nstvout         = 1000
nstenergy       = 1000
nstlog          = 1000

; Temperature coupling
tcoupl          = V-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = {temperature}

; Pressure coupling. C-rescale rather than Parrinello-Rahman: the box starts far
; from equilibrium here, where Parrinello-Rahman can oscillate.
pcoupl          = C-rescale
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {pressure}
compressibility = 4.5e-5
refcoord-scaling = all

; Constraints
constraints     = {constraint_type}
constraint_algorithm = lincs

; Continue from NVT velocities and constraint state. The matching checkpoint is
; supplied to grompp when it is available.
continuation    = yes
gen_vel         = no

{get_cutoff_mdp_section(force_field)}
"""

MACE_OFF_PAIR_CUTOFF_NM: float = 0.5


def get_nnpot_model_input_mdp_section(model_name: str) -> str:
    """Return the GROMACS input contract required by a wrapped NNP model."""
    if model_name not in NNPOT_MODEL_PACKAGES:
        raise ValueError(
            f"Unsupported NNPot model {model_name!r}. Choose one of: "
            + ", ".join(SUPPORTED_NNPOT_MODELS)
        )

    if model_name.startswith("mace-"):
        # All MACE-OFF small/medium/large foundation models have r_max=5 Å.
        # Let GROMACS build the neighbour list: its pair shifts correctly cover
        # triclinic boxes, partial PBC and periodic images without an O(N^2)
        # all-pairs allocation in the TorchScript wrapper.
        return f"""nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = nnp-charge
nnpot-model-input4    = atom-pairs
nnpot-model-input5    = pair-shifts
nnpot-model-input6    = box
nnpot-model-input7    = pbc
pair-cutoff            = {MACE_OFF_PAIR_CUTOFF_NM}"""

    if model_name == "ani2x-emle":
        # EMLE is an electrostatic embedding model, not merely an ANI alias.  It
        # receives the MM environment and returns forces on both regions.
        return """nnpot-embedding       = electrostatic-model
nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = atom-positions-mm
nnpot-model-input4    = atom-charges-mm
nnpot-model-input5    = nnp-charge
nnpot-model-input6    = box"""

    if model_name == "aimnet2":
        # Charge comes from the selected topology group rather than a baked-in
        # neutral default in the cached model.
        return """nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = nnp-charge
nnpot-model-input4    = box
nnpot-model-input5    = pbc"""

    return """nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = nnp-charge
nnpot-model-input4    = box
nnpot-model-input5    = pbc"""


def get_default_prod_md_mdp_file_content(time_scale_ps: float = 1000, time_step_ps: float = 0.002,
                                         temperature: float = 300, pressure: float = 1.0,
                                         mdp_type: str = "Initial", random_seed: int = 0,
                                         with_ligand: bool = False, nnpot_active: bool = False,
                                         nnpot_modelfile_path: str | None = "models/ani2x.pt",
                                         nnpot_input_group: str = "Protein",
                                         nnpot_model_name: str = "ani2x",
                                         force_field: str | None = None) -> str:
    """MDP for unrestrained production MD, optionally driven by a neural potential."""
    _, time_step_ps, nsteps = _simulation_step_count(time_scale_ps, time_step_ps)
    temperature = _positive_finite_number(temperature, "Temperature")
    pressure = _positive_finite_number(pressure, "Pressure")
    constraint_type = get_dynamics_constraint_type(force_field)
    content = f"""
integrator      = md
dt              = {time_step_ps}
nsteps          = {nsteps}

; Output
nstxout         = 0
nstvout         = 0
nstenergy       = 5000
nstlog          = 5000
nstxout-compressed = 5000

{get_cutoff_mdp_section(force_field)}

; Temperature coupling
tcoupl          = V-rescale
tc-grps         = System
tau_t           = 0.1
ref_t           = {temperature}

; Constraints
constraints     = {constraint_type}
constraint_algorithm = lincs
"""
    if nnpot_active:
        content = content + """
; Pressure coupling
; NNPot wrappers currently return energies, not virials/stress. Keep the box
; fixed after NPT equilibration to avoid unstable pressure scaling.
pcoupl          = no
"""
    else:
        content = content + f"""
; Pressure coupling
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {pressure}
compressibility = 4.5e-5
"""
    # Production always starts from an equilibrated state. Generating another
    # Maxwell distribution here would discard the NPT velocities, even for the
    # first production segment. ``mdp_type`` and ``random_seed`` remain in the
    # public signature for compatibility with saved Gradio configurations.
    content = content + """
; Continue from the equilibrated coordinates, velocities and coupling state
continuation    = yes
gen_vel         = no
"""

    if nnpot_active:
        nnpot_input_group = _single_line_configuration_value(
            nnpot_input_group, "NNPot input group")
        nnpot_modelfile_path = _single_line_configuration_value(
            nnpot_modelfile_path, "NNPot model file")
        content = content + "\n; Neural network potential (machine learning interatomic potential)\n"
        content = content + "nnpot-active          = true\n"
        content = content + f"nnpot-modelfile       = {nnpot_modelfile_path}\n"
        content = content + f"nnpot-input-group     = {nnpot_input_group}\n"
        content = content + get_nnpot_model_input_mdp_section(nnpot_model_name)

    return content


def get_matching_checkpoint_path(working_directory_path: str,
                                 input_structure_file_name: str) -> str | None:
    """Return ``<input stem>.cpt`` when present, without following an escape symlink."""
    checkpoint_name = os.path.splitext(input_structure_file_name)[0] + ".cpt"
    checkpoint_path = validate_local_file_path(
        working_directory_path, checkpoint_name, "matching checkpoint file")
    return checkpoint_path if os.path.isfile(checkpoint_path) else None


def require_matching_resume_files(working_directory_path: str,
                                  run_input_file_name: str | None,
                                  checkpoint_file_name: str | None) -> tuple[str, str]:
    """Validate the TPR/checkpoint pair used to resume an interrupted run.

    ``mdrun -cpi`` resumes the remaining steps encoded in an existing TPR; it
    does not extend that TPR.  Keeping both files on one ``-deffnm`` stem avoids
    accidentally appending a foreign checkpoint to another run's outputs.
    """
    run_input_path = validate_local_file_path(
        working_directory_path, run_input_file_name, "run input file")
    checkpoint_path = validate_local_file_path(
        working_directory_path, checkpoint_file_name, "checkpoint file")
    # ``validate_local_file_path`` rejects missing/blank names.  Keep concrete
    # strings below without relying on assertions, which disappear under
    # ``python -O``.
    if not isinstance(run_input_file_name, str) or not run_input_file_name.strip():
        raise ValueError("Select a production .tpr run input file.")
    if not isinstance(checkpoint_file_name, str) or not checkpoint_file_name.strip():
        raise ValueError("Select a production .cpt checkpoint file.")

    if os.path.splitext(run_input_file_name)[1].lower() != ".tpr":
        raise ValueError("Production resume requires a .tpr run input file.")
    if os.path.splitext(checkpoint_file_name)[1].lower() != ".cpt":
        raise ValueError("Production resume requires a .cpt checkpoint file.")
    run_stem = os.path.splitext(run_input_file_name)[0]
    checkpoint_stem = os.path.splitext(checkpoint_file_name)[0]
    if run_stem != checkpoint_stem:
        raise ValueError(
            f"Checkpoint '{checkpoint_file_name}' does not match run input "
            f"'{run_input_file_name}'. Select '{run_stem}.cpt'."
        )
    if not os.path.isfile(run_input_path):
        raise ValueError(f"Run input file '{run_input_file_name}' does not exist.")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint file '{checkpoint_file_name}' does not exist.")
    return run_input_file_name, checkpoint_file_name

LIGAND_RESNAME: str = "LIG"

# gmx_MMPBSA runs in the job directory, because a topology's #include lines only
# resolve beside it, and leaves dozens of these behind. They are working files,
# not results, so they are hidden the same way GROMACS backups are. Both tabs
# browse the same directories, so both have to hide them.
MMPBSA_SCRATCH_PREFIX: str = "_GMXMMPBSA_"

# Columns 18-20 of a PDB ATOM/HETATM/TER record, 1-based and inclusive.
_PDB_RESNAME_SLICE = slice(17, 20)

BLANK_RESNAME_LABEL: str = "(blank)"

def read_pdb_residue_names(pdb_file_path: str) -> list[str]:
    """Distinct residue names on the ATOM/HETATM records, in the order met.

    A record with nothing in columns 18-20 is reported as BLANK_RESNAME_LABEL:
    real files do come with the field empty, and it has to be visible rather
    than silently treated as "no residues here".
    """
    names: list[str] = []
    with open(pdb_file_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            name = line.rstrip("\r\n")[_PDB_RESNAME_SLICE].strip() or BLANK_RESNAME_LABEL
            if name not in names:
                names.append(name)

    return names

def rename_pdb_residues(pdb_file_path: str, resname: str = LIGAND_RESNAME,
                        source_resname: str | None = None) -> list[str]:
    """Rewrite residue names in a PDB file in place, returning the old names.

    An uploaded ligand often calls its molecule UNK, MOL, a PDB chemical
    component id, or nothing at all. Trajectory analysis selects the ligand as
    "resname LIG", so anything else silently analyses an empty selection.

    ``source_resname`` restricts the rewrite to one residue name, for a file that
    holds more than the ligand. When it is None, or names nothing in the file,
    every ATOM/HETATM record is rewritten - an uploaded ligand file is the
    ligand. Returns the replaced names, empty when nothing needed changing.
    """
    with open(pdb_file_path) as handle:
        lines = handle.readlines()

    present = read_pdb_residue_names(pdb_file_path)
    selective = bool(source_resname) and source_resname in present

    replaced: list[str] = []
    rewritten: list[str] = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM", "TER")):
            rewritten.append(line)
            continue

        body = line.rstrip("\r\n")
        newline = line[len(body):]
        current = body[_PDB_RESNAME_SLICE].strip()

        # A bare "TER" carries no residue name and is left alone. An ATOM or
        # HETATM with the field empty is a different matter: it still needs a
        # name, and skipping it was why a blank-resname ligand reached acpype
        # unnamed and came back as UNK.
        if line.startswith("TER") and not current:
            rewritten.append(line)
            continue
        if selective and current != source_resname:
            rewritten.append(line)
            continue

        label = current or BLANK_RESNAME_LABEL
        if current != resname and label not in replaced:
            replaced.append(label)

        rewritten.append(body[:_PDB_RESNAME_SLICE.start] + resname.rjust(3)
                         + body[_PDB_RESNAME_SLICE.stop:] + newline)

    if not replaced:
        return []

    with open(pdb_file_path, "w") as handle:
        handle.writelines(rewritten)

    return replaced

# Grace commands in an .xvg header. Every gmx analysis tool writes these, so the
# axis labels and per-series legends come from the file rather than from hardcoded
# per-tool knowledge.
_XVG_TITLE = re.compile(r'^@\s+title\s+"(.*)"')
_XVG_AXIS_LABEL = re.compile(r'^@\s+(x|y)axis\s+label\s+"(.*)"')
_XVG_LEGEND = re.compile(r'^@\s+s(\d+)\s+legend\s+"(.*)"')

def read_xvg(xvg_file_path: str) -> XvgData:
    """Parse a GROMACS .xvg into a DataFrame whose columns are its own legends.

    The format is '#' comments, '@' grace commands carrying the title, axis labels
    and one legend per series, then whitespace-separated numbers. Files written
    with -xvg none carry no header at all, so every label is optional.
    """
    title = ""
    xlabel = ""
    ylabel = ""
    legends: dict[int, str] = {}
    rows: list[list[float]] = []

    with open(xvg_file_path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("@"):
                match = _XVG_LEGEND.match(line)
                if match:
                    legends[int(match.group(1))] = match.group(2)
                    continue
                match = _XVG_AXIS_LABEL.match(line)
                if match:
                    if match.group(1) == "x":
                        xlabel = match.group(2)
                    else:
                        ylabel = match.group(2)
                    continue
                match = _XVG_TITLE.match(line)
                if match:
                    title = match.group(1)
                continue

            try:
                rows.append([float(value) for value in line.split()])
            except ValueError:
                # '&' separates datasets in multi-set files. Anything else that does
                # not parse is not data either, so skip it rather than fail outright.
                continue

    if not rows:
        raise ValueError(f"{os.path.basename(xvg_file_path)} contains no data rows.")

    # A run killed mid-write leaves a short final line; keep only the full-width rows
    # so one truncated line cannot turn the whole frame into NaNs.
    width = len(rows[0])
    rows = [row for row in rows if len(row) == width]

    columns = [xlabel or "x"]
    for index in range(width - 1):
        # A two-column file usually names its single series only on the y axis.
        default = ylabel if width == 2 and ylabel else f"y{index}"
        columns.append(legends.get(index) or default)

    # Duplicate legends happen (gmx sasa -or repeats a label); pandas would allow the
    # collision but every lookup by name would then return a frame, not a series.
    seen: dict[str, int] = {}
    for index, name in enumerate(columns):
        if name in seen:
            seen[name] += 1
            columns[index] = f"{name} ({seen[name]})"
        else:
            seen[name] = 1

    return XvgData(frame=pd.DataFrame(rows, columns=columns), title=title,
                   xlabel=xlabel, ylabel=ylabel)

def make_line_figure(frame: pd.DataFrame, x_column: str | None = None,
                     y_columns: Sequence[str] | None = None, xlabel: str | None = None,
                     ylabel: str | None = None, title: str | None = None,
                     mean_line: bool = False) -> Any:
    """Plot columns of a frame onto a standalone matplotlib Figure.

    Deliberately not plt.figure(): the pyplot API draws into a process-wide "current
    figure", so two analyses running at once in the Gradio worker would draw into
    each other. A bare Figure has no global state. Gradio encodes it just the same.
    """
    # Local import: utils is imported directly by the tests, and pulling matplotlib
    # in at module scope would make them depend on a writable cache directory.
    from matplotlib.figure import Figure

    x_column = x_column if x_column is not None else frame.columns[0]
    if y_columns is None:
        y_columns = [column for column in frame.columns if column != x_column]

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    for column in y_columns:
        axes.plot(frame[x_column], frame[column], label=column)

    if mean_line and len(y_columns) == 1:
        mean = frame[y_columns[0]].mean()
        axes.axhline(mean, color="red", linestyle="--", label=f"Mean {y_columns[0]}")

    axes.set_xlabel(xlabel if xlabel is not None else x_column)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title:
        axes.set_title(title)
    if len(y_columns) > 1 or mean_line:
        axes.legend()

    figure.tight_layout()
    return figure


def gromacs_backbone_fitted_rmsd(
        working_directory_path: str,
        run_input_file_name: str,
        input_traj_file_name: str,
        measurement_groups: Sequence[str],
        *,
        group_resolver: Callable[[Sequence[str], Sequence[str], str], str],
        command_runner: Callable[..., Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return topology-aware RMSD time and group series from ``gmx rms``.

    ``gmx rms`` makes molecules whole using the TPR before applying one shared
    backbone fit.  Requesting all measurement groups in the same invocation is
    important for a ligand: fitting it independently would erase motion relative
    to the protein.  Values are returned in the units used by this UI (ns and A).
    """
    if not measurement_groups:
        raise ValueError("At least one RMSD measurement group is required.")

    directory = validate_working_directory(working_directory_path)
    descriptor, output_path = tempfile.mkstemp(
        prefix=".rmsd_", suffix=".xvg", dir=directory)
    os.close(descriptor)
    os.unlink(output_path)
    output_name = os.path.basename(output_path)

    cmd = [
        "gmx", "rms",
        "-s", run_input_file_name,
        "-f", input_traj_file_name,
        "-o", output_name,
        "-tu", "ns",
        "-pbc", "yes",
        "-fit", "rot+trans",
        "-mw", "yes",
        "-ng", str(len(measurement_groups)),
        "-xvg", "xmgrace",
    ]
    try:
        group_input = group_resolver(
            cmd, ["Backbone", *measurement_groups], directory)
        command_runner(cmd, cwd=directory, stdin_input=group_input)
        parsed = read_xvg(output_path)
        values = parsed["frame"].to_numpy(dtype=float)

        expected_columns = len(measurement_groups) + 1
        if values.ndim != 2 or values.shape[1] != expected_columns:
            raise ValueError(
                "gmx rms returned an unexpected number of data columns: "
                f"expected {expected_columns}, found "
                f"{values.shape[1] if values.ndim == 2 else 'invalid data'}.")
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("gmx rms returned empty or non-finite data.")

        xlabel = parsed["xlabel"].lower().replace(" ", "")
        ylabel = parsed["ylabel"].lower().replace(" ", "")
        if "time" not in xlabel or "ns" not in xlabel:
            raise ValueError(
                "gmx rms output did not identify its time axis in nanoseconds.")
        if "rmsd" not in ylabel or "nm" not in ylabel:
            raise ValueError(
                "gmx rms output did not identify RMSD values in nanometres.")

        times_ns = values[:, 0]
        rmsd_nm = values[:, 1:]
        if np.any(np.diff(times_ns) < 0):
            raise ValueError("gmx rms returned a decreasing time axis.")
        if np.any(rmsd_nm < 0):
            raise ValueError("gmx rms returned a negative RMSD value.")
        rmsd_angstrom = rmsd_nm * 10.0
        if not np.all(np.isfinite(rmsd_angstrom)):
            raise ValueError("Converted RMSD values are not finite.")
        return times_ns.copy(), rmsd_angstrom
    finally:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass


def ca_residue_metadata(ca_atoms: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return stable plot indices and unique, chain-aware C-alpha labels."""
    raw_labels: list[str] = []
    for atom in ca_atoms:
        try:
            chain = str(atom.chainID).strip()
        except Exception:
            chain = ""
        if not chain:
            try:
                segment = str(atom.segid).strip()
                chain = "" if segment.upper() == "SYSTEM" else segment
            except Exception:
                chain = ""
        residue = f"{atom.resname}{atom.resid}"
        raw_labels.append(f"{chain}:{residue}" if chain else residue)

    totals: dict[str, int] = {}
    for label in raw_labels:
        totals[label] = totals.get(label, 0) + 1
    seen: dict[str, int] = {}
    labels: list[str] = []
    for label in raw_labels:
        seen[label] = seen.get(label, 0) + 1
        labels.append(
            label if totals[label] == 1 else f"{label} [{seen[label]}]")

    return (np.arange(1, len(raw_labels) + 1, dtype=int),
            np.asarray(labels, dtype=object))


def _valid_periodic_dimensions(dimensions: Any) -> np.ndarray | None:
    """Return a usable triclinic unit cell, or None for a non-periodic frame."""
    try:
        box = np.asarray(dimensions, dtype=float)
    except (TypeError, ValueError):
        return None
    if (box.shape != (6,) or not np.all(np.isfinite(box))
            or np.any(box[:3] <= 0) or np.any(box[3:] <= 0)
            or np.any(box[3:] >= 180)):
        return None
    return box


def _make_protein_fragments_whole_and_clustered(
        protein: Any, fragments: Sequence[Any], dimensions: Any) -> None:
    """Place bonded protein chains in one periodic image in the current frame."""
    box = _valid_periodic_dimensions(dimensions)
    if box is None:
        return

    # Connectivity from the TPR is what makes this scientifically safer than a
    # coordinate-only atom-order heuristic.  Each bonded chain is made whole.
    protein.unwrap(compound="fragments", reference=None, inplace=True)
    if len(fragments) < 2:
        return

    # Keep the largest chain fixed and put every other chain in its nearest image.
    # This avoids the wrapped/multi-chain jump that a plain backbone fit cannot
    # remove while preserving each chain's internal conformation.
    anchor = max(fragments, key=lambda group: group.n_atoms)
    anchor_center = np.asarray(anchor.center_of_geometry(), dtype=float)
    if not np.all(np.isfinite(anchor_center)):
        raise ValueError("Protein chain coordinates are not finite.")
    for fragment in fragments:
        if fragment is anchor:
            continue
        center = np.asarray(fragment.center_of_geometry(), dtype=float)
        displacement = center - anchor_center
        minimum_image = np.asarray(
            _minimize_vectors(displacement.reshape(1, 3), box),
            dtype=float)[0]
        fragment.translate(anchor_center + minimum_image - center)


MAX_RMSF_CHUNK_COORDINATES = 2_000_000
MAX_RMSF_CHUNK_FRAMES = 200


def _remove_temporary_trajectory_bundle(trajectory_path: str) -> None:
    """Remove a temporary XTC and MDAnalysis' colocated offset cache files."""
    directory = os.path.dirname(trajectory_path)
    basename = os.path.basename(trajectory_path)
    for path in (
            trajectory_path,
            os.path.join(directory, f".{basename}_offsets.npz"),
            os.path.join(directory, f".{basename}_offsets.lock")):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def gromacs_topology_aware_ca_rmsf(
        working_directory_path: str,
        run_input_file_name: str,
        input_traj_file_name: str,
        structure_file_name: str,
        *,
        group_resolver: Callable[[Sequence[str], Sequence[str], str], str],
        command_runner: Callable[..., Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C-alpha RMSF after bounded native-GROMACS PBC clustering.

    Current GROMACS TPR revisions can be newer than MDAnalysis' TPR parser.
    Native ``gmx trjconv -pbc cluster`` therefore supplies whole, clustered
    protein coordinates in fixed-size chunks.  Each chunk is deleted after a
    streaming backbone fit and Welford update, bounding both RAM and disk use.
    """
    directory = validate_working_directory(working_directory_path)
    source: mda.Universe | None = None
    protein_universe: mda.Universe | None = None
    temporary_output_path: str | None = None
    try:
        source = mda.Universe(
            os.path.join(directory, structure_file_name),
            os.path.join(directory, input_traj_file_name))
        total_frames = len(source.trajectory)
        if total_frames == 0:
            raise ValueError("The trajectory contains no frames.")
        source_protein = source.select_atoms("protein")
        source_ca = source.select_atoms("protein and name CA")
        if source_protein.n_atoms == 0 or source_ca.n_atoms == 0:
            raise ValueError("No protein C-alpha atoms were found in the structure.")

        protein_universe = mda.Merge(source_protein)
        protein = protein_universe.atoms
        backbone = protein_universe.select_atoms("protein and backbone")
        ca_atoms = protein_universe.select_atoms("protein and name CA")
        if backbone.n_atoms < 3:
            raise ValueError(
                "At least three protein backbone atoms are required to align the trajectory.")
        if ca_atoms.n_atoms != source_ca.n_atoms:
            raise ValueError("Protein atom ordering changed while preparing RMSF analysis.")
        plot_indices, labels = ca_residue_metadata(source_ca)

        weights = np.asarray(backbone.masses, dtype=float)
        if (weights.shape != (backbone.n_atoms,)
                or not np.all(np.isfinite(weights)) or np.any(weights <= 0)):
            weights = np.ones(backbone.n_atoms, dtype=float)

        frames_per_chunk = max(1, min(
            MAX_RMSF_CHUNK_FRAMES,
            MAX_RMSF_CHUNK_COORDINATES // max(1, protein.n_atoms)))
        mean = np.zeros((ca_atoms.n_atoms, 3), dtype=float)
        sum_squares = np.zeros_like(mean)
        frame_count = 0
        reference_center: np.ndarray | None = None
        reference_centered: np.ndarray | None = None
        previous_time = -math.inf
        group_input: str | None = None

        for start in range(0, total_frames, frames_per_chunk):
            stop = min(total_frames, start + frames_per_chunk) - 1
            start_time = float(source.trajectory[start].time)
            stop_time = float(source.trajectory[stop].time)
            if (not math.isfinite(start_time) or not math.isfinite(stop_time)
                    or stop_time < start_time):
                raise ValueError("The trajectory has an invalid or decreasing time axis.")

            descriptor, temporary_output_path = tempfile.mkstemp(
                prefix=".rmsf_cluster_", suffix=".xtc", dir=directory)
            os.close(descriptor)
            os.unlink(temporary_output_path)
            output_name = os.path.basename(temporary_output_path)
            cmd = [
                "gmx", "trjconv",
                "-s", run_input_file_name,
                "-f", input_traj_file_name,
                "-o", output_name,
                "-pbc", "cluster",
                "-ur", "compact",
                "-b", format(start_time, ".17g"),
                "-e", format(stop_time, ".17g"),
                "-tu", "ps",
            ]
            if group_input is None:
                # trjconv asks first for the cluster group and then the output
                # group (or vice versa in some releases); both are Protein.
                group_input = group_resolver(
                    cmd, ["Protein", "Protein"], directory)
            command_runner(cmd, cwd=directory, stdin_input=group_input)

            protein_universe.load_new(temporary_output_path)
            expected_frames = stop - start + 1
            if len(protein_universe.trajectory) != expected_frames:
                raise ValueError(
                    "GROMACS returned an unexpected number of RMSF chunk frames: "
                    f"expected {expected_frames}, found "
                    f"{len(protein_universe.trajectory)}.")
            if protein.n_atoms != source_protein.n_atoms:
                raise ValueError(
                    "The GROMACS Protein group does not match the selected structure.")

            for timestep in protein_universe.trajectory:
                time_ps = float(timestep.time)
                if not math.isfinite(time_ps) or time_ps < previous_time:
                    raise ValueError(
                        "GROMACS returned an invalid or decreasing trajectory time axis.")
                previous_time = time_ps
                mobile_backbone = np.asarray(backbone.positions, dtype=float)
                ca_positions = np.asarray(ca_atoms.positions, dtype=float)
                if (not np.all(np.isfinite(mobile_backbone))
                        or not np.all(np.isfinite(ca_positions))):
                    raise ValueError(
                        "The clustered trajectory contains non-finite coordinates.")
                mobile_center = np.average(
                    mobile_backbone, axis=0, weights=weights)
                if reference_centered is None:
                    reference_center = mobile_center.copy()
                    reference_centered = (
                        mobile_backbone - reference_center).copy()
                rotation, _ = _rotation_matrix(
                    mobile_backbone - mobile_center,
                    reference_centered, weights=weights)
                fitted_ca = (np.dot(ca_positions - mobile_center, rotation.T)
                             + reference_center)
                if not np.all(np.isfinite(fitted_ca)):
                    raise ValueError(
                        "Backbone fitting produced non-finite coordinates.")

                frame_count += 1
                delta = fitted_ca - mean
                mean += delta / frame_count
                sum_squares += delta * (fitted_ca - mean)

            protein_universe.trajectory.close()
            _remove_temporary_trajectory_bundle(temporary_output_path)
            temporary_output_path = None

        if frame_count != total_frames:
            raise ValueError(
                f"RMSF processed {frame_count} of {total_frames} trajectory frames.")
        rmsf = np.sqrt(np.maximum(
            np.sum(sum_squares / frame_count, axis=1), 0.0))
        if not np.all(np.isfinite(rmsf)):
            raise ValueError("Calculated C-alpha RMSF values are not finite.")
        return plot_indices, labels, rmsf
    finally:
        if protein_universe is not None:
            protein_universe.trajectory.close()
        if temporary_output_path is not None:
            _remove_temporary_trajectory_bundle(temporary_output_path)
        if source is not None:
            source.trajectory.close()


def topology_aware_ca_rmsf(
        topology_file_path: str,
        trajectory_file_path: str,
        label_structure_file_path: str | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate PBC-corrected, backbone-aligned C-alpha RMSF in one pass.

    The TPR supplies molecular bonds.  Protein fragments are unwrapped and
    multi-chain assemblies are clustered by minimum image before fitting every
    frame to frame zero.  Welford accumulation retains only O(number of C-alpha
    atoms) coordinates regardless of trajectory length.
    """
    universe: mda.Universe | None = None
    label_universe: mda.Universe | None = None
    try:
        universe = mda.Universe(topology_file_path, trajectory_file_path)
        if len(universe.trajectory) == 0:
            raise ValueError("The trajectory contains no frames.")

        protein = universe.select_atoms("protein")
        backbone = universe.select_atoms("protein and backbone")
        ca_atoms = universe.select_atoms("protein and name CA")
        if protein.n_atoms == 0 or ca_atoms.n_atoms == 0:
            raise ValueError("No protein C-alpha atoms were found in the TPR.")
        if backbone.n_atoms < 3:
            raise ValueError(
                "At least three protein backbone atoms are required to align the trajectory.")
        try:
            fragments = tuple(protein.fragments)
        except Exception as exc:
            raise ValueError(
                "The selected topology does not provide protein bond connectivity; "
                "choose the production TPR that generated this trajectory.") from exc
        if not fragments:
            raise ValueError("The TPR contains no bonded protein fragments.")

        weights = np.asarray(backbone.masses, dtype=float)
        if (weights.shape != (backbone.n_atoms,)
                or not np.all(np.isfinite(weights)) or np.any(weights <= 0)):
            weights = np.ones(backbone.n_atoms, dtype=float)

        universe.trajectory[0]
        _make_protein_fragments_whole_and_clustered(
            protein, fragments, universe.dimensions)
        reference_backbone = np.asarray(backbone.positions, dtype=float).copy()
        if not np.all(np.isfinite(reference_backbone)):
            raise ValueError("The reference backbone contains non-finite coordinates.")
        reference_center = np.average(reference_backbone, axis=0, weights=weights)
        reference_centered = reference_backbone - reference_center

        if label_structure_file_path:
            label_universe = mda.Universe(label_structure_file_path)
            label_ca = label_universe.select_atoms("protein and name CA")
            if label_ca.n_atoms != ca_atoms.n_atoms:
                raise ValueError(
                    "The selected structure and TPR contain different numbers of "
                    "protein C-alpha atoms.")
            plot_indices, labels = ca_residue_metadata(label_ca)
        else:
            plot_indices, labels = ca_residue_metadata(ca_atoms)

        mean = np.zeros((ca_atoms.n_atoms, 3), dtype=float)
        sum_squares = np.zeros_like(mean)
        frame_count = 0
        previous_time = -math.inf
        for timestep in universe.trajectory:
            time_ps = float(timestep.time)
            if not math.isfinite(time_ps):
                raise ValueError("The trajectory contains a non-finite frame time.")
            if time_ps < previous_time:
                raise ValueError("The trajectory time axis decreases between frames.")
            previous_time = time_ps

            _make_protein_fragments_whole_and_clustered(
                protein, fragments, timestep.dimensions)
            mobile_backbone = np.asarray(backbone.positions, dtype=float)
            ca_positions = np.asarray(ca_atoms.positions, dtype=float)
            if (not np.all(np.isfinite(mobile_backbone))
                    or not np.all(np.isfinite(ca_positions))):
                raise ValueError("The trajectory contains non-finite protein coordinates.")
            mobile_center = np.average(mobile_backbone, axis=0, weights=weights)
            rotation, _ = _rotation_matrix(
                mobile_backbone - mobile_center, reference_centered,
                weights=weights)
            fitted_ca = np.dot(ca_positions - mobile_center, rotation.T) + reference_center
            if not np.all(np.isfinite(fitted_ca)):
                raise ValueError("Backbone fitting produced non-finite coordinates.")

            frame_count += 1
            delta = fitted_ca - mean
            mean += delta / frame_count
            sum_squares += delta * (fitted_ca - mean)

        if frame_count == 0:
            raise ValueError("The trajectory contains no readable frames.")
        rmsf = np.sqrt(np.maximum(
            np.sum(sum_squares / frame_count, axis=1), 0.0))
        if not np.all(np.isfinite(rmsf)):
            raise ValueError("Calculated C-alpha RMSF values are not finite.")
        return plot_indices, labels, rmsf
    finally:
        if label_universe is not None:
            label_universe.trajectory.close()
        if universe is not None:
            universe.trajectory.close()


def backbone_aligned_ca_rmsf(
        universe: mda.Universe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return plot indices, unique residue labels, and aligned C-alpha RMSF.

    The reference contains only one frame of backbone coordinates.  The input
    trajectory itself remains file-backed and the alignment is applied as each
    frame is read, which avoids copying a production trajectory into memory.
    """
    ca_atoms = universe.select_atoms("protein and name CA")
    if ca_atoms.n_atoms == 0:
        raise Exception("No C-alpha atoms found. Is this a protein structure?")

    backbone = universe.select_atoms("protein and backbone")
    if backbone.n_atoms < 3:
        raise Exception("At least three protein backbone atoms are required to align the trajectory.")

    # Merge copies only the currently loaded backbone coordinates into a
    # one-frame MemoryReader, leaving the (potentially multi-GB) trajectory on
    # disk.  Pin the current frame explicitly so the first trajectory frame is
    # always the fit reference.
    universe.trajectory[0]
    reference = mda.Merge(backbone)
    try:
        universe.trajectory.add_transformations(
            _fit_rot_trans(backbone, reference.atoms)
        )
        values = _mda_rms.RMSF(ca_atoms).run().results.rmsf.copy()
        plot_indices, labels = ca_residue_metadata(ca_atoms)
    finally:
        reference.trajectory.close()

    return plot_indices, labels, values


def periodic_center_of_mass(atom_group: Any, box: Any) -> np.ndarray:
    """Return a mass-weighted COM without splitting a molecule across PBC.

    Coordinates are unwrapped by accumulating minimum-image displacements
    between consecutive atoms.  Using the atom ordering as a continuity path is
    important for proteins wider than half a box: mapping every atom directly
    to the image nearest atom zero folds an already-whole extended molecule and
    silently moves its COM.  The returned point can lie outside the primary
    unit cell; callers comparing two centres should minimum-image the final
    centre-to-centre displacement.  With absent or invalid box dimensions, this
    has exactly the ordinary MDAnalysis COM behaviour.
    """
    ordinary_com = np.asarray(atom_group.center_of_mass(), dtype=float)
    try:
        dimensions = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return ordinary_com

    if (dimensions.shape != (6,) or not np.all(np.isfinite(dimensions))
            or np.any(dimensions[:3] <= 0)
            or np.any(dimensions[3:] <= 0)
            or np.any(dimensions[3:] >= 180)):
        return ordinary_com

    positions = np.asarray(atom_group.positions, dtype=float)
    masses = np.asarray(atom_group.masses, dtype=float)
    if (positions.ndim != 2 or positions.shape[0] == 0
            or positions.shape != (masses.size, 3)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(masses))
            or masses.sum() <= 0):
        return ordinary_com

    try:
        from MDAnalysis.lib.distances import minimize_vectors
        consecutive_displacements = minimize_vectors(
            positions[1:] - positions[:-1], dimensions)
    except (TypeError, ValueError):
        return ordinary_com

    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    if positions.shape[0] > 1:
        unwrapped[1:] = positions[0] + np.cumsum(
            consecutive_displacements, axis=0)
    return np.average(unwrapped, axis=0, weights=masses)

def format_running_status(cmd: Sequence[str], note: str = "") -> str:
    """The status shown while a command is still running.

    Escaped, because selection strings reach here straight from a textbox and
    GROMACS syntax is full of characters that would otherwise be read as HTML.
    """
    import html

    message = f"Running command:<br><code>{html.escape(' '.join(str(part) for part in cmd))}</code>"
    if note:
        message = f"{html.escape(note)}<br>{message}"

    return f"<span style='color:orange;'>{message}</span>"

# idecomp, as gmx_MMPBSA numbers the schemes. 1 and 2 are per-residue; 3 and 4
# are pairwise, which produces a residue-by-residue matrix and is far slower.
MMPBSA_DECOMPOSITION_SCHEMES: tuple[tuple[str, int], ...] = (
    ("Per-residue, 1-4 terms added to internal", 1),
    ("Per-residue, 1-4 terms added to EEL/VDW", 2),
    ("Pairwise, 1-4 terms added to internal", 3),
    ("Pairwise, 1-4 terms added to EEL/VDW", 4),
)

def get_default_mmpbsa_input_file_content(start_frame: int = 1, end_frame: int = 0,
                                          interval: int = 1, salt_concentration: float = 0.150,
                                          temperature: float = 300.0, use_gb: bool = True,
                                          use_pb: bool = False, gb_model: int = 2,
                                          use_decomposition: bool = True,
                                          decomposition_scheme: int = 2,
                                          print_residues: str = "within 6",
                                          protein_force_field: str | None = None,
                                          ligand_force_field: str | None = None) -> str:
    """The &general/&gb/&pb/&decomp namelists gmx_MMPBSA reads from its -i file.

    endframe = 0 means "to the end of the trajectory" here; gmx_MMPBSA wants a
    real frame number, so it is only written when the caller gives one.

    The &decomp namelist is what makes the per-residue contributions appear;
    without it gmx_MMPBSA reports only the total binding energy.
    """
    if not use_gb and not use_pb:
        raise ValueError("Select at least one of Generalised Born or Poisson-Boltzmann.")

    try:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        interval = int(interval)
        decomposition_scheme = int(decomposition_scheme)
    except (TypeError, ValueError):
        raise ValueError("MM-PBSA frame and decomposition settings must be whole numbers.") from None
    if start_frame < 1:
        raise ValueError("MM-PBSA start frame must be at least 1.")
    if end_frame < 0 or (end_frame and end_frame < start_frame):
        raise ValueError("MM-PBSA end frame must be 0 or no earlier than the start frame.")
    if interval < 1:
        raise ValueError("MM-PBSA frame interval must be at least 1.")
    if decomposition_scheme not in {1, 2, 3, 4}:
        raise ValueError("MM-PBSA decomposition scheme must be 1, 2, 3, or 4.")

    temperature = _positive_finite_number(temperature, "MM-PBSA temperature")
    try:
        salt_concentration = float(salt_concentration)
    except (TypeError, ValueError):
        raise ValueError("MM-PBSA salt concentration must be a number.") from None
    if not math.isfinite(salt_concentration) or salt_concentration < 0:
        raise ValueError("MM-PBSA salt concentration must be finite and non-negative.")

    print_residues = _single_line_configuration_value(
        print_residues, "MM-PBSA residue selection")
    if '"' in print_residues:
        raise ValueError("MM-PBSA residue selection cannot contain a quote.")

    content = ("Input file generated by GROMACS WebUI\n"
               "&general\n"
               f"  startframe        = {start_frame},\n")
    if end_frame > 0:
        content += f"  endframe          = {end_frame},\n"
    content += (f"  interval          = {interval},\n"
                f"  temperature       = {temperature},\n")
    # The WebUI always launches gmx_MMPBSA with ``-cp`` and therefore supplies
    # the actual GROMACS topology.  Its documentation says ``forcefields`` is
    # unnecessary in this mode; writing unrelated ff14SB/GAFF defaults can be
    # misleading and, in topology-less modes, scientifically inconsistent.
    if protein_force_field or ligand_force_field:
        if not (protein_force_field and ligand_force_field):
            raise ValueError(
                "Both protein and ligand force fields must be supplied together."
            )
        protein_force_field = _single_line_configuration_value(
            protein_force_field, "Protein force field")
        ligand_force_field = _single_line_configuration_value(
            ligand_force_field, "Ligand force field")
        if '"' in protein_force_field or '"' in ligand_force_field:
            raise ValueError("Force-field names cannot contain a quote.")
        content += (f"  forcefields       = \"{protein_force_field}\", "
                    f"\"{ligand_force_field}\",\n")
    content += ("  sys_name          = \"Protein-ligand complex\",\n"
                "  keep_files        = 0,\n"
                "  verbose           = 2,\n"
                "/\n")

    if use_gb:
        content += ("&gb\n"
                    f"  igb               = {int(gb_model)},\n"
                    f"  saltcon           = {salt_concentration},\n"
                    "/\n")
    if use_pb:
        content += ("&pb\n"
                    f"  istrng            = {salt_concentration},\n"
                    "  inp               = 2,\n"
                    "  radiopt           = 0,\n"
                    "/\n")

    if use_decomposition:
        # dec_verbose = 3 prints the per-residue breakdown for the complex, the
        # receptor, the ligand and the delta; the delta is the one worth reading,
        # and the others make the difference checkable.
        content += ("&decomp\n"
                    f"  idecomp           = {decomposition_scheme},\n"
                    "  dec_verbose       = 3,\n"
                    f"  print_res         = \"{print_residues}\",\n"
                    "  csv_format        = 1,\n"
                    "/\n")

    return content

# The five statistics gmx_MMPBSA prints per term, in the order of its own header:
# "Energy Component  Average  SD(Prop.)  SD  SEM(Prop.)  SEM".
MMPBSA_STATISTIC_COLUMNS: tuple[str, ...] = ("Average (kcal/mol)", "SD(Prop.)", "SD",
                                             "SEM(Prop.)", "SEM")

# gmx_MMPBSA repeats the whole report once per method it was asked for, headed by
# these lines. The first entry doubles as the label for a report with no header
# at all, which is what a single-method run of an older version writes.
MMPBSA_METHOD_NAMES: dict[str, str] = {
    "GENERALIZED BORN": "GB",
    "POISSON BOLTZMANN": "PB",
    "GBNSR6": "GBNSR6",
    "3D-RISM": "RISM",
    "NMODE": "NMODE",
    "QUASI-HARMONIC APPROXIMATION": "QH",
}

_MMPBSA_DECOMPOSITION_METHOD_NAMES: dict[str, str] = {
    "GENERALIZED BORN DECOMPOSITION ENERGIES": "GB",
    "POISSON BOLTZMANN DECOMPOSITION ENERGIES": "PB",
    "GENERALIZED BORN (R6) DECOMPOSITION ENERGIES": "GBNSR6",
}


def _mmpbsa_method_name(line: str) -> str | None:
    """Return the calculation method named by a gmx_MMPBSA section heading.

    The summary and the two companion CSV files use different headings for the
    same methods.  A decomposition heading can also be quoted across two lines
    because it contains an embedded newline in gmx_MMPBSA's CSV writer.
    """
    heading = line.strip().strip('"').strip().rstrip(":").upper()
    if heading in MMPBSA_METHOD_NAMES:
        return MMPBSA_METHOD_NAMES[heading]
    return _MMPBSA_DECOMPOSITION_METHOD_NAMES.get(heading)


def _split_mmpbsa_method_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split a companion CSV into its GB/PB/etc. calculation sections.

    Older single-method files have no method heading.  They retain the historic
    GB label for backwards compatibility, while current files get one section
    per heading so a dual GB/PB run cannot silently lose its second half.
    """
    sections: list[tuple[str, list[str]]] = []
    method: str | None = None
    start = 0
    for index, line in enumerate(lines):
        next_method = _mmpbsa_method_name(line)
        if next_method is None:
            continue
        if method is not None:
            sections.append((method, lines[start:index]))
        method = next_method
        start = index + 1

    if method is not None:
        sections.append((method, lines[start:]))
        return sections

    return [(next(iter(MMPBSA_METHOD_NAMES.values())), lines)]

def _is_number(text: str) -> bool:
    """Whether a whitespace-separated field parses as a float."""
    try:
        float(text)
    except ValueError:
        return False

    return True

def parse_mmpbsa_results(dat_file_path: str) -> pd.DataFrame:
    """Pull the energy decomposition out of a FINAL_RESULTS_MMPBSA.dat file.

    The file is a human-readable report rather than a table: several sections,
    each with a header line then "TERM  average  sd  sd(mean)" rows. Only the
    delta sections matter.

    A run that asked for both MM-GBSA and MM-PBSA writes the whole report twice,
    once under "GENERALIZED BORN:" and once under "POISSON BOLTZMANN:", each
    with its own delta section and the same term names. Every row therefore
    carries the method it came from, so the two cannot be mistaken for one set
    of duplicated terms.
    """
    with open(dat_file_path) as handle:
        lines = handle.readlines()

    rows: list[dict[str, Any]] = []
    in_delta = False
    method = next(iter(MMPBSA_METHOD_NAMES.values()))
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        method_heading = _mmpbsa_method_name(stripped)
        if method_heading is not None:
            method = method_heading
            in_delta = False
            continue

        fields = stripped.split()
        # One section per component, headed by its name: "Complex:", "Receptor:",
        # "Ligand:", "Delta (Complex - Receptor - Ligand):". Those same words also
        # begin summary rows carrying numbers, so a heading is the one with none.
        section = fields[0].rstrip(":").lower()
        if section in ("complex", "receptor", "ligand", "delta"):
            if not any(_is_number(field) for field in fields[1:]):
                in_delta = section == "delta"
                continue

        if not in_delta:
            continue

        # A term row is a name followed by its statistics. Split from the right
        # because the name itself can contain a space ("Δ1-4 VDW"), and take the
        # count from the row rather than assuming it: a GB run and a PB run print
        # different terms, and the column header line has no numbers at all.
        numbers: list[float] = []
        while len(numbers) < len(fields) and _is_number(fields[len(fields) - 1 - len(numbers)]):
            numbers.insert(0, float(fields[len(fields) - 1 - len(numbers)]))
        if not numbers or len(numbers) == len(fields):
            continue

        row: dict[str, Any] = {"Term": " ".join(fields[:len(fields) - len(numbers)]),
                               "Method": method}
        for column, value in zip(MMPBSA_STATISTIC_COLUMNS, numbers):
            row[column] = value
        rows.append(row)

    if not rows:
        raise ValueError(f"No energy decomposition found in {os.path.basename(dat_file_path)}. "
                         f"Open it in the text viewer to see what gmx_MMPBSA reported.")

    return pd.DataFrame(rows)

def _read_mmpbsa_csv_block(lines: list[str], header_index: int) -> pd.DataFrame:
    """Read one "Frame #,..." table out of a gmx_MMPBSA CSV, stopping at its end.

    These files hold several tables separated by blank lines and section titles,
    so a plain read_csv would swallow the lot.
    """
    header = [field.strip() for field in lines[header_index].strip().split(",")]
    rows: list[list[str]] = []
    for line in lines[header_index + 1:]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != len(header):
            break
        rows.append(fields)

    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if not converted.isna().all():
            frame[column] = converted

    return frame

def _find_mmpbsa_section(lines: list[str], *titles: str) -> int:
    """Index of the header row belonging to the last of the given section titles.

    Titles are matched in order, so ("DELTAS:", "Total Decomposition") finds the
    delta section's own table rather than the complex's table of the same name.
    """
    position = 0
    for title in titles:
        for index in range(position, len(lines)):
            if lines[index].strip().startswith(title):
                position = index + 1
                break
        else:
            raise ValueError(f"No '{title}' section found.")

    for index in range(position - 1, len(lines)):
        if lines[index].strip().startswith("Frame #"):
            return index

    raise ValueError("No data table follows the section title.")

def parse_mmpbsa_per_frame(csv_file_path: str) -> pd.DataFrame:
    """The binding energy of each individual frame, from gmx_MMPBSA's -eo file.

    That is the "Delta Energy Terms" table: complex minus receptor minus ligand,
    one row per frame, which is what a distribution of the binding energy is
    built from.
    """
    with open(csv_file_path) as handle:
        lines = handle.readlines()

    frames: list[pd.DataFrame] = []
    section_errors: list[ValueError] = []
    for method, section_lines in _split_mmpbsa_method_sections(lines):
        try:
            header_index = _find_mmpbsa_section(section_lines, "Delta Energy Terms")
        except ValueError as exc:
            # A method heading can precede a partial/failed section.  Do not let
            # it hide a complete method later in the same output file.
            section_errors.append(exc)
            continue
        frame = _read_mmpbsa_csv_block(section_lines, header_index)
        if not frame.empty:
            frame.insert(1, "Method", method)
            frames.append(frame)

    if not frames:
        if section_errors:
            raise section_errors[0]
        raise ValueError(f"{os.path.basename(csv_file_path)} holds no per-frame energies.")

    return pd.concat(frames, ignore_index=True, sort=False)

def parse_mmpbsa_decomposition(csv_file_path: str) -> pd.DataFrame:
    """Per-residue contributions to the binding energy, averaged over frames.

    Read from the DELTAS section of gmx_MMPBSA's -deo file, which reports every
    printed residue for every frame; the mean is the contribution and the spread
    across frames says how steady it is.
    """
    with open(csv_file_path) as handle:
        lines = handle.readlines()

    frames: list[pd.DataFrame] = []
    section_errors: list[ValueError] = []
    for method, section_lines in _split_mmpbsa_method_sections(lines):
        try:
            header_index = _find_mmpbsa_section(
                section_lines, "DELTAS:", "Total Decomposition Contribution")
        except ValueError as exc:
            section_errors.append(exc)
            continue
        per_frame = _read_mmpbsa_csv_block(section_lines, header_index)
        if per_frame.empty:
            continue

        if "Residue" not in per_frame:
            raise ValueError(
                f"{os.path.basename(csv_file_path)} contains pairwise decomposition data; "
                "choose a per-residue decomposition scheme to use this panel."
            )

        value_columns = [c for c in per_frame.columns
                         if c not in ("Frame #", "Residue")
                         and per_frame[c].dtype.kind in "if"]
        grouped = per_frame.groupby("Residue", sort=False)
        frame = grouped[value_columns].mean().reset_index()
        frame["TOTAL SD"] = (grouped["TOTAL"].std().to_numpy()
                             if "TOTAL" in value_columns else float("nan"))
        frame.insert(1, "Method", method)
        frames.append(frame)

    if not frames:
        if section_errors:
            raise section_errors[0]
        raise ValueError(f"{os.path.basename(csv_file_path)} holds no decomposition data.")

    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.sort_values("TOTAL").reset_index(drop=True) if "TOTAL" in result else result

# gmx_MMPBSA labels a receptor residue "R:A:LEU:37" and a ligand one "L:B:LIG:245".
MMPBSA_LIGAND_RESIDUE_PREFIX: str = "L:"
MMPBSA_LIGAND_BAR_COLOUR: str = "tab:orange"
MMPBSA_RECEPTOR_BAR_COLOUR: str = "tab:blue"

def mmpbsa_residue_colours(residues: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Bar colours that separate the ligand's own term from the receptor residues.

    The ligand contributes most of the binding energy almost by definition, so
    left in one colour it simply dwarfs the residues you are trying to read.
    """
    colours = [MMPBSA_LIGAND_BAR_COLOUR if str(name).startswith(MMPBSA_LIGAND_RESIDUE_PREFIX)
               else MMPBSA_RECEPTOR_BAR_COLOUR for name in residues]
    legend = {}
    if MMPBSA_RECEPTOR_BAR_COLOUR in colours:
        legend[MMPBSA_RECEPTOR_BAR_COLOUR] = "Receptor residue"
    if MMPBSA_LIGAND_BAR_COLOUR in colours:
        legend[MMPBSA_LIGAND_BAR_COLOUR] = "Ligand"

    return colours, legend

def read_mmpbsa_frame_selection(input_file_path: str) -> tuple[int, int]:
    """The startframe and interval an mmpbsa.in asked for, defaulting to 1 and 1.

    gmx_MMPBSA numbers its own frames 1..N over the frames it selected, so these
    two are what map a result back onto the trajectory it came from.
    """
    start_frame, interval = 1, 1
    with open(input_file_path) as handle:
        for line in handle:
            match = re.match(r"\s*(startframe|interval)\s*=\s*(\d+)", line)
            if match:
                if match.group(1) == "startframe":
                    start_frame = int(match.group(2))
                else:
                    interval = int(match.group(2))

    return start_frame, interval

def get_trajectory_frame_times_ns(structure_file_path: str, trajectory_file_path: str,
                                  start_frame: int, interval: int, count: int) -> list[float]:
    """Simulation time, in ns, of the frames a gmx_MMPBSA run actually used.

    Read from the trajectory rather than assumed from a constant step, so a
    trajectory that was concatenated or written unevenly still lines up.
    """
    universe = mda.Universe(structure_file_path, trajectory_file_path)
    try:
        total = len(universe.trajectory)
        times: list[float] = []
        for number in range(count):
            index = (start_frame - 1) + number * interval
            if index >= total:
                break
            times.append(universe.trajectory[index].time / 1000)
        return times
    finally:
        universe.trajectory.close()

def make_histogram_figure(values: Any, bins: int = 30, xlabel: str = "", title: str = "") -> Any:
    """Distribution of a per-frame quantity, with its mean marked."""
    from matplotlib.figure import Figure

    series = np.asarray(values, dtype=float)
    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    axes.hist(series, bins=bins, color="tab:blue", edgecolor="white")
    axes.axvline(series.mean(), color="red", linestyle="--",
                 label=f"Mean {series.mean():.2f}")

    axes.set_xlabel(xlabel)
    axes.set_ylabel("Frames")
    if title:
        axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    return figure

def make_bar_figure(frame: pd.DataFrame, label_column: str, value_column: str,
                    error_column: str | None = None, ylabel: str = "",
                    title: str = "", colors: Sequence[str] | None = None,
                    legend: dict[str, str] | None = None) -> Any:
    """Bar chart of an energy decomposition, with error bars when available.

    ``colors`` overrides the default colouring by sign, for when the bars are
    grouped by something more interesting than whether they are positive.
    ``legend`` maps a colour to its meaning and is drawn with proxy handles,
    since the bars themselves carry no labels.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    errors = frame[error_column] if error_column and error_column in frame else None
    axes.bar(frame[label_column], frame[value_column], yerr=errors, capsize=4,
             color=list(colors) if colors is not None else
             ["tab:red" if value > 0 else "tab:blue" for value in frame[value_column]])
    axes.axhline(0, color="black", linewidth=0.8)
    if legend:
        axes.legend(handles=[Patch(facecolor=colour, label=text)
                             for colour, text in legend.items()])

    axes.set_ylabel(ylabel or value_column)
    if title:
        axes.set_title(title)
    axes.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure

# Gas constant in the units GROMACS reports energies in.
BOLTZMANN_CONSTANT_KJ_PER_MOL_K: float = 0.008314462618

def compute_free_energy_landscape(x_values: Any, y_values: Any, bin_count: int = 100,
                                  temperature: float = 300.0) -> tuple[Any, Any, Any, Any]:
    """Turn a 2D sample into a Gibbs free energy surface.

    G = -kT ln(P / P_max), so the most populated bin sits at exactly 0 and every
    other bin is positive: a depth below the deepest well, which is what a free
    energy landscape is read as. Bins nothing ever visited are NaN rather than
    +inf, because matplotlib leaves NaN blank but +inf would flatten the colour
    scale onto a single level.

    Returns (x_centres, y_centres, probability, free_energy), the two grids
    indexed [x, y] as numpy.histogram2d returns them.
    """
    try:
        bins = int(bin_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Bin count must be an integer from 2 to 1000.") from exc
    if isinstance(bin_count, (float, np.floating)) and not float(bin_count).is_integer():
        raise ValueError("Bin count must be an integer from 2 to 1000.")
    if not 2 <= bins <= 1000:
        raise ValueError("Bin count must be an integer from 2 to 1000.")

    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Temperature must be a positive finite number.") from exc
    if not math.isfinite(temperature_value) or temperature_value <= 0:
        raise ValueError("Temperature must be a positive finite number.")

    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(y_values, dtype=float).reshape(-1)
    if x_array.size != y_array.size:
        raise ValueError("The two projection axes must contain the same number of points.")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("Projection coordinates must contain only finite values.")

    counts, x_edges, y_edges = np.histogram2d(x_array, y_array, bins=bins)
    if counts.sum() == 0:
        raise ValueError("The projection contains no points to build a landscape from.")

    probability = counts / counts.sum()
    free_energy = np.full(probability.shape, np.nan)
    populated = probability > 0
    free_energy[populated] = (-BOLTZMANN_CONSTANT_KJ_PER_MOL_K * temperature_value
                              * np.log(probability[populated] / probability.max()))

    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2

    return x_centres, y_centres, probability, free_energy

def make_landscape_figure(x_centres: Any, y_centres: Any, free_energy: Any,
                          xlabel: str = "", ylabel: str = "", title: str = "",
                          levels: int = 30) -> Any:
    """Filled contour plot of a free energy surface, with its minimum marked."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    # histogram2d indexes [x, y] but contourf reads [row, column] as [y, x], so the
    # grid is transposed here. Without this the landscape is silently mirrored
    # about the diagonal and still looks entirely plausible.
    contours = axes.contourf(x_centres, y_centres, free_energy.T, levels=levels, cmap="viridis")
    figure.colorbar(contours, ax=axes, label="ΔG (kJ/mol)")

    deepest = np.unravel_index(np.nanargmin(free_energy), free_energy.shape)
    axes.plot(x_centres[deepest[0]], y_centres[deepest[1]], "wx", markersize=12,
              markeredgewidth=2, label="Minimum")
    axes.legend(loc="upper right")

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title:
        axes.set_title(title)
    figure.tight_layout()
    return figure

def make_scree_figure(frame: pd.DataFrame, count: int = 20, title: str = "") -> Any:
    """Eigenvalue bars with the cumulative share of the variance over them."""
    from matplotlib.figure import Figure

    if frame.shape[1] < 2 or frame.empty:
        raise ValueError("The eigenvalue table must contain at least one row and two columns.")
    try:
        shown_count = int(count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scree component count must be a positive integer.") from exc
    if isinstance(count, (float, np.floating)) and not float(count).is_integer():
        raise ValueError("Scree component count must be a positive integer.")
    if shown_count < 1:
        raise ValueError("Scree component count must be a positive integer.")

    all_eigenvalues = frame.iloc[:, 1].to_numpy(dtype=float)
    if not np.isfinite(all_eigenvalues).all():
        raise ValueError("Eigenvalues must contain only finite values.")
    total_variance = all_eigenvalues.sum()
    if not math.isfinite(float(total_variance)) or total_variance <= 0:
        raise ValueError("Eigenvalues must have a positive total variance.")

    indices = frame.iloc[:shown_count, 0].to_numpy()
    eigenvalues = all_eigenvalues[:shown_count]
    cumulative = np.cumsum(eigenvalues) / total_variance * 100

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    axes.bar(indices, eigenvalues, color="tab:blue", label="Eigenvalue")
    axes.set_xlabel(frame.columns[0])
    axes.set_ylabel(f"Eigenvalue {frame.columns[1]}")

    share = axes.twinx()
    share.plot(indices, cumulative, color="tab:red", marker="o", label="Cumulative variance")
    share.set_ylabel("Cumulative variance (%)")
    share.set_ylim(0, 100)

    if title:
        axes.set_title(title)
    figure.tight_layout()
    return figure

def make_scatter_figure(frame: pd.DataFrame, xlabel: str = "", ylabel: str = "",
                        title: str = "", colour_label: str = "Frame") -> Any:
    """Scatter of the first two columns, coloured by row order."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    points = axes.scatter(frame.iloc[:, 0], frame.iloc[:, 1],
                          c=np.arange(len(frame)), cmap="viridis", s=12)
    figure.colorbar(points, ax=axes, label=colour_label)

    axes.set_xlabel(xlabel or frame.columns[0])
    axes.set_ylabel(ylabel or frame.columns[1])
    if title:
        axes.set_title(title)
    figure.tight_layout()
    return figure

def read_gromacs_structure_file(filename: str) -> tuple[str, int, list[str], str]:
    """Split a GRO file into its title, atom count, atom lines and box line."""
    with open(filename) as f:
        lines = f.readlines()

    display_name = os.path.basename(filename)
    if len(lines) < 3:
        raise ValueError(f"GRO file '{display_name}' is truncated (expected title, atom count, and box).")

    title = lines[0].strip()
    try:
        natoms = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"GRO file '{display_name}' has an invalid atom count.") from exc
    if natoms < 0:
        raise ValueError(f"GRO file '{display_name}' has a negative atom count.")
    if len(lines) < natoms + 3:
        raise ValueError(
            f"GRO file '{display_name}' is truncated: it declares {natoms} atoms "
            f"but does not contain all atom records and a box line."
        )

    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms].strip()
    for atom_index, atom_line in enumerate(atoms, start=1):
        # GRO coordinates occupy fixed-width columns 21-44. Checking those
        # fields catches incomplete uploads before a malformed complex is saved.
        if len(atom_line.rstrip("\r\n")) < 44:
            raise ValueError(
                f"GRO file '{display_name}' has a truncated atom record at line {atom_index + 2}."
            )
        try:
            coordinates = [float(atom_line[start:start + 8]) for start in (20, 28, 36)]
        except ValueError as exc:
            raise ValueError(
                f"GRO file '{display_name}' has invalid coordinates at line {atom_index + 2}."
            ) from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"GRO file '{display_name}' has non-finite coordinates at line {atom_index + 2}."
            )

    try:
        box_values = [float(value) for value in box.split()]
    except ValueError as exc:
        raise ValueError(f"GRO file '{display_name}' has an invalid box line.") from exc
    if len(box_values) not in (3, 9) or not all(math.isfinite(value) for value in box_values):
        raise ValueError(
            f"GRO file '{display_name}' must have three or nine finite box values."
        )

    return title, natoms, atoms, box

LIGAND_CLASH_DISTANCE_NM = 0.05
LIGAND_FAR_DISTANCE_WARNING_NM = 5.0
_MAX_STRUCTURE_DISTANCE_PAIRS = 1_000_000


def _gro_atom_identity(atom_line: str) -> tuple[str, str]:
    """Return the residue and atom names from one validated fixed-width GRO row."""
    return atom_line[5:10].strip(), atom_line[10:15].strip()


def _gro_coordinates(atom_lines: Sequence[str]) -> np.ndarray:
    """Return an ``(n, 3)`` coordinate array from validated GRO atom rows."""
    return np.asarray([
        [float(line[start:start + 8]) for start in (20, 28, 36)]
        for line in atom_lines
    ], dtype=float)


def _minimum_cross_structure_distance(first: np.ndarray,
                                      second: np.ndarray) -> float:
    """Find the minimum cross-set distance without allocating an unbounded matrix."""
    if not len(first) or not len(second):
        return math.inf

    block_size = max(1, int(math.sqrt(_MAX_STRUCTURE_DISTANCE_PAIRS)))
    minimum_squared = math.inf
    for first_start in range(0, len(first), block_size):
        first_block = first[first_start:first_start + block_size]
        second_block_size = max(
            1, _MAX_STRUCTURE_DISTANCE_PAIRS // max(1, len(first_block)))
        for second_start in range(0, len(second), second_block_size):
            second_block = second[second_start:second_start + second_block_size]
            deltas = first_block[:, None, :] - second_block[None, :, :]
            block_minimum = float(np.min(np.einsum(
                "ijk,ijk->ij", deltas, deltas)))
            minimum_squared = min(minimum_squared, block_minimum)
            if minimum_squared < LIGAND_CLASH_DISTANCE_NM ** 2:
                return math.sqrt(minimum_squared)
    return math.sqrt(minimum_squared)


def _read_ligand_itp_atoms(
        ligand_topology_file_path: str) -> list[tuple[str, str]]:
    """Read ordered ``(residue name, atom name)`` identities from one ligand ITP.

    A structure merge can only represent one molecule instance. Conditional or
    otherwise non-literal ``[ atoms ]`` contents cannot be paired confidently,
    so those inputs fail with an actionable message instead of being guessed.
    """
    display_name = os.path.basename(ligand_topology_file_path)
    atoms: list[tuple[str, str]] = []
    in_atoms = False
    found_atoms_section = False
    with open(ligand_topology_file_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            section_name = _gromacs_section_name(line)
            if section_name is not None:
                if in_atoms:
                    break
                in_atoms = section_name == "atoms"
                found_atoms_section = found_atoms_section or in_atoms
                continue
            if not in_atoms:
                continue

            content = line.split(";", 1)[0].strip()
            if not content:
                continue
            if content.startswith("#"):
                raise ValueError(
                    f"Ligand topology '{display_name}' has preprocessor logic in "
                    "its [ atoms ] section, so its coordinate ordering cannot be "
                    "verified safely. Flatten that section before merging."
                )
            tokens = content.split()
            if len(tokens) < 5:
                raise ValueError(
                    f"Ligand topology '{display_name}' has an invalid [ atoms ] "
                    f"entry: '{content}'."
                )
            try:
                atom_number = int(tokens[0])
            except ValueError as exc:
                raise ValueError(
                    f"Ligand topology '{display_name}' has a non-integer atom "
                    f"number in its [ atoms ] section: '{tokens[0]}'."
                ) from exc
            expected_number = len(atoms) + 1
            if atom_number != expected_number:
                raise ValueError(
                    f"Ligand topology '{display_name}' has atom number "
                    f"{atom_number} where {expected_number} was expected; its "
                    "coordinate ordering cannot be verified safely."
                )
            atoms.append((tokens[3], tokens[4]))

    if not found_atoms_section or not atoms:
        raise ValueError(
            f"Ligand topology '{display_name}' has no non-empty [ atoms ] section, "
            "so it cannot be paired with ligand coordinates."
        )
    return atoms


def _acpype_pair_stem(file_path: str, extension: str) -> str | None:
    """Return the stem from an ACPYPE ``<stem>_GMX.<ext>`` output name."""
    file_name = os.path.basename(file_path)
    ending = f"_GMX{extension}"
    if not file_name.lower().endswith(ending.lower()):
        return None
    return file_name[:-len(ending)].casefold()


def _existing_acpype_counterpart(file_path: str, source_extension: str,
                                 counterpart_extension: str) -> str | None:
    """Return an existing sibling from the same canonical ACPYPE output set."""
    if _acpype_pair_stem(file_path, source_extension) is None:
        return None
    counterpart_path = file_path[:-len(source_extension)] + counterpart_extension
    return counterpart_path if os.path.isfile(counterpart_path) else None


def validate_ligand_gro_itp_pair(ligand_structure_file_path: str,
                                 ligand_topology_file_path: str) -> None:
    """Verify that a ligand GRO and ITP describe the same ordered atoms.

    GROMACS binds coordinates to topology atoms by position, not by atom name.
    Comparing the ordered identities prevents a same-sized but crossed pair from
    silently assigning coordinates to the wrong atom types.
    """
    gro_stem = _acpype_pair_stem(ligand_structure_file_path, ".gro")
    itp_stem = _acpype_pair_stem(ligand_topology_file_path, ".itp")
    if gro_stem is not None and itp_stem is not None and gro_stem != itp_stem:
        raise ValueError(
            "The selected ligand coordinates and topology come from different "
            "ACPYPE output sets "
            f"('{os.path.basename(ligand_structure_file_path)}' versus "
            f"'{os.path.basename(ligand_topology_file_path)}')."
        )

    _, gro_count, gro_lines, _ = read_gromacs_structure_file(
        ligand_structure_file_path)
    gro_atoms = [_gro_atom_identity(line) for line in gro_lines]
    topology_atoms = _read_ligand_itp_atoms(ligand_topology_file_path)
    if gro_count != len(topology_atoms):
        raise ValueError(
            "The selected ligand coordinate/topology pair has different atom "
            f"counts ({gro_count} in {os.path.basename(ligand_structure_file_path)}, "
            f"{len(topology_atoms)} in {os.path.basename(ligand_topology_file_path)})."
        )

    for atom_index, (gro_atom, topology_atom) in enumerate(
            zip(gro_atoms, topology_atoms), start=1):
        if tuple(value.casefold() for value in gro_atom) != tuple(
                value.casefold() for value in topology_atom):
            gro_residue, gro_name = gro_atom
            topology_residue, topology_name = topology_atom
            raise ValueError(
                "The selected ligand coordinate/topology pair has different "
                f"ordered atoms at position {atom_index}: GRO "
                f"{gro_residue}:{gro_name}, ITP "
                f"{topology_residue}:{topology_name}. Select files generated "
                "from the same ligand."
            )


def merge_protein_ligand_structures(
        protein_structure_file_path: str, ligand_structure_file_path: str,
        output_structure_file_path: str,
        ligand_topology_file_path: str | None = None) -> list[str]:
    """Concatenate protein and ligand coordinates, keeping the protein box.

    Returns non-fatal placement warnings for display by the WebUI.
    """
    # Read input files
    _, p_n, p_atoms, p_box = read_gromacs_structure_file(protein_structure_file_path)
    _, l_n, l_atoms, _ = read_gromacs_structure_file(ligand_structure_file_path)
    ligand_topology_file_path = (
        ligand_topology_file_path
        or _existing_acpype_counterpart(
            ligand_structure_file_path, ".gro", ".itp"))
    if ligand_topology_file_path is not None:
        validate_ligand_gro_itp_pair(
            ligand_structure_file_path, ligand_topology_file_path)

    if p_n == 0 or l_n == 0:
        empty_component = "protein" if p_n == 0 else "ligand"
        raise ValueError(
            f"Cannot merge a structure with an empty {empty_component} component."
        )

    minimum_distance = _minimum_cross_structure_distance(
        _gro_coordinates(p_atoms), _gro_coordinates(l_atoms))
    if minimum_distance < LIGAND_CLASH_DISTANCE_NM:
        raise ValueError(
            "Protein and ligand contain catastrophically overlapping atoms "
            f"(minimum separation {minimum_distance:.3f} nm). Check that both "
            "files use nanometres and the same coordinate frame."
        )

    warnings: list[str] = []
    if minimum_distance > LIGAND_FAR_DISTANCE_WARNING_NM:
        warnings.append(
            "The ligand is unusually far from the protein "
            f"(minimum separation {minimum_distance:.2f} nm). Verify that the "
            "docked pose and protein use the same coordinate frame before "
            "continuing."
        )

    # Combination
    total_atoms = p_n + l_n

    # Assemble the validated structure before replacing the destination.  A
    # failed/short write must not destroy an earlier usable complex.gro.
    content = (
        "Protein + ligand complex\n"
        f"{total_atoms}\n"
        + "".join(p_atoms)
        + "".join(l_atoms)
        # Keep the protein box (the ligand box is meaningless on its own).
        + p_box + "\n"
    )
    atomic_write_text_file(output_structure_file_path, content)
    return warnings

_GROMACS_SECTION_RE = re.compile(r"^\s*\[\s*([^]]+?)\s*]\s*(?:;.*)?$")
_GROMACS_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+["<]([^">]+)[">]')
_LIGAND_BLOCK_BEGIN = "; GROMACS WebUI ligand topology begin"
_LIGAND_BLOCK_END = "; GROMACS WebUI ligand topology end"
_LIGAND_MOLECULE_NAME_PREFIX = "; GROMACS WebUI ligand molecule type: "
_LIGAND_MOLECULE_BEGIN = "; GROMACS WebUI ligand molecule begin"
_LIGAND_MOLECULE_END = "; GROMACS WebUI ligand molecule end"


def _gromacs_section_name(line: str) -> str | None:
    """Return a normalised GROMACS section name, if *line* starts a section."""
    match = _GROMACS_SECTION_RE.match(line)
    return match.group(1).strip().lower() if match else None


def _gromacs_include_name(line: str) -> str | None:
    """Return the path named by a GROMACS preprocessor include."""
    match = _GROMACS_INCLUDE_RE.match(line)
    return match.group(1) if match else None


def get_topology_force_field_name(topology_file_path: str) -> str | None:
    """Return the ``*.ff`` directory name used by a GROMACS topology.

    A topology may contain many includes, so only the canonical
    ``<name>.ff/forcefield.itp`` include identifies the selected force field.
    Multiple different force-field includes are rejected rather than choosing
    whichever one happened to appear first.
    """
    names: list[str] = []
    with open(topology_file_path, "r") as topology_file:
        for line in topology_file:
            include_name = _gromacs_include_name(line)
            if include_name is None:
                continue
            components = include_name.replace("\\", "/").split("/")
            if len(components) < 2 or components[-1].lower() != "forcefield.itp":
                continue
            directory_name = components[-2]
            if directory_name.lower().endswith(".ff"):
                names.append(directory_name[:-3])

    distinct = list(dict.fromkeys(name.lower() for name in names))
    if len(distinct) > 1:
        raise ValueError(
            f"Topology '{os.path.basename(topology_file_path)}' includes multiple "
            f"force fields: {', '.join(names)}."
        )
    return names[0] if names else None


def get_topology_force_field_family(topology_file_path: str) -> str | None:
    """Return AMBER, CHARMM, GROMOS or OPLS for a recognised topology include."""
    return get_force_field_family(get_topology_force_field_name(topology_file_path))


GROMOS_SINGLE_RANGE_WARNING = (
    "The selected GROMOS force field was parameterized with the historical "
    "twin-range scheme. GROMACS warns that modern single-range/Verlet results "
    "can differ from the intended values; verify that this model is appropriate "
    "for your system. No other grompp warning was bypassed."
)


def run_grompp_with_gromos_warning_policy(
        cmd: Sequence[str], cwd: str, topology_file_path: str,
        max_warnings: int, runner: Any = None) -> str | None:
    """Run grompp, narrowly allowing its unavoidable modern GROMOS warning.

    Current GROMACS releases emit one warning for bundled GROMOS force fields
    because the original twin-range algorithm is no longer available.  Leaving
    ``-maxwarn 0`` makes every advertised GROMOS workflow unusable; blindly
    setting it to one can hide an unrelated topology or physics warning.  This
    helper permits exactly the documented GROMOS warning, rejects every other
    warning, and returns text that callers must surface as an orange status.
    User-requested non-zero ``max_warnings`` values retain their existing expert
    override semantics and are not inspected here.
    """
    if runner is None:
        runner = run_checked_command

    effective_cmd = list(cmd)
    automatic_allowance = (
        max_warnings == 0
        and get_topology_force_field_family(topology_file_path) == "GROMOS"
    )
    if not automatic_allowance:
        runner(effective_cmd, cwd=cwd)
        return None

    try:
        maxwarn_index = effective_cmd.index("-maxwarn") + 1
    except (ValueError, IndexError):
        raise ValueError("Internal error: grompp command has no -maxwarn value.") \
            from None
    effective_cmd[maxwarn_index] = "1"
    try:
        output_index = effective_cmd.index("-o") + 1
        final_output_path = effective_cmd[output_index]
    except (ValueError, IndexError):
        raise ValueError("Internal error: grompp command has no -o value.") \
            from None
    try:
        processed_index = effective_cmd.index("-po") + 1
        final_processed_path = effective_cmd[processed_index]
    except (ValueError, IndexError):
        processed_index = None
        final_processed_path = None

    # grompp writes its TPR before returning success.  Use disposable paths so
    # an unexpected allowed warning can never destroy or leave behind a runnable
    # version in place of the user's last known-good output.
    with tempfile.TemporaryDirectory(prefix=".grompp_stage_", dir=cwd) as stage:
        staged_output_path = os.path.join(
            stage, os.path.basename(final_output_path))
        effective_cmd[output_index] = staged_output_path
        staged_processed_path = None
        if processed_index is not None and final_processed_path is not None:
            staged_processed_path = os.path.join(
                stage, os.path.basename(final_processed_path))
            effective_cmd[processed_index] = staged_processed_path

        process = runner(effective_cmd, cwd=cwd)
        stderr = getattr(process, "stderr", "")
        stdout = getattr(process, "stdout", "")
        output = ((stderr if isinstance(stderr, str) else "") + "\n"
                  + (stdout if isinstance(stdout, str) else ""))
        warning_blocks = _extract_gromacs_warning_blocks(output)
        summary_counts = [
            int(value) for value in re.findall(
                r"(?im)^\s*There (?:was|were)\s+(\d+)\s+WARNING(?:S)?\s*$",
                output)
        ]
        warning_count = max(summary_counts, default=len(warning_blocks))

        warning_text = "\n".join(warning_blocks).lower()
        allowance_unused = warning_count == 0 and not warning_blocks
        expected = (
            warning_count == 1
            and len(warning_blocks) == 1
            and "the gromos force fields have been parametrized" in warning_text
            and "twin-range cut-off" in warning_text
            and "single-range cut-off" in warning_text
        )
        if not (allowance_unused or expected):
            observed = "\n\n".join(warning_blocks) or (
                f"GROMACS reported {warning_count} warning(s).")
            raise ValueError(
                "GROMACS emitted a warning that is not covered by the narrow "
                "GROMOS compatibility allowance. No new run input was kept:\n"
                + observed
            )

        if not os.path.isfile(staged_output_path):
            raise FileNotFoundError(
                "grompp reported success but did not create its run input file."
            )
        if (staged_processed_path is not None
                and not os.path.isfile(staged_processed_path)):
            raise FileNotFoundError(
                "grompp reported success but did not create its processed MDP file."
            )

        # If another session starts a writer in the tiny hand-off interval after
        # grompp exits, this lease refuses publication instead of changing files
        # underneath it.  The TPR is replaced last so it is never paired with an
        # older processed-MDP snapshot after an interrupted publish.
        with reserve_working_directory_maintenance(cwd):
            if (staged_processed_path is not None
                    and final_processed_path is not None):
                os.replace(staged_processed_path, final_processed_path)
            os.replace(staged_output_path, final_output_path)

    return GROMOS_SINGLE_RANGE_WARNING if expected else None


def validate_topology_force_field(topology_file_path: str,
                                  selected_force_field: str | None) -> str:
    """Validate a UI selection against the topology and return its actual name.

    Variants within one family share the cutoff policy used by this application,
    so selecting one AMBER variant while loading another is accepted here. The
    exact topology name is returned for downstream MDP generation.
    """
    detected = get_topology_force_field_name(topology_file_path)
    if detected is None:
        raise ValueError(
            f"Topology '{os.path.basename(topology_file_path)}' has no "
            "<force-field>.ff/forcefield.itp include."
        )

    if selected_force_field is None or not str(selected_force_field).strip():
        return detected

    detected_family = get_force_field_family(detected)
    selected_family = get_force_field_family(selected_force_field)
    same_assumptions = (
        detected_family == selected_family
        if detected_family is not None or selected_family is not None
        else detected.strip().lower() == str(selected_force_field).strip().lower()
    )
    if not same_assumptions:
        raise ValueError(
            f"Selected force field '{selected_force_field}' does not match topology "
            f"force field '{detected}'."
        )
    return detected


def _read_mdp_settings(mdp_file_path: str) -> dict[str, str]:
    """Read the last effective value of each simple ``key = value`` MDP entry."""
    settings: dict[str, str] = {}
    with open(mdp_file_path, "r") as mdp_file:
        for line in mdp_file:
            content = line.split(";", 1)[0]
            if "=" not in content:
                continue
            key, value = content.split("=", 1)
            normalised_key = re.sub(r"[-_]", "", key.strip().lower())
            if normalised_key:
                settings[normalised_key] = value.strip()
    return settings


def validate_mdp_topology_compatibility(mdp_file_path: str,
                                        topology_file_path: str) -> str:
    """Validate family-sensitive MDP cutoffs against the actual topology.

    Returns the force-field name from the topology. Family-specific cutoff and
    electrostatics rules are checked without requiring byte-for-byte generated
    defaults; unknown custom families retain only format/numeric sanity checks.
    """
    force_field = get_topology_force_field_name(topology_file_path)
    if force_field is None:
        raise ValueError(
            f"Topology '{os.path.basename(topology_file_path)}' has no "
            "<force-field>.ff/forcefield.itp include."
        )
    family = get_force_field_family(force_field)
    settings = _read_mdp_settings(mdp_file_path)
    problems: list[str] = []

    def numeric_setting(name: str, default: float | None = None,
                        *, positive: bool = False) -> float | None:
        raw_value = settings.get(name)
        if raw_value is None:
            return default
        try:
            value = float(raw_value.split()[0])
        except (IndexError, ValueError):
            problems.append(f"{name} has invalid value '{raw_value}'")
            return None
        if not math.isfinite(value) or (positive and value <= 0):
            requirement = "a finite positive number" if positive else "a finite number"
            problems.append(f"{name} must be {requirement}, not '{raw_value}'")
            return None
        return value

    time_step = numeric_setting("dt", positive=True)
    cutoff_values: dict[str, float | None] = {}
    for name in ("rlist", "rvdw", "rcoulomb"):
        # Custom families are still checked for malformed numeric syntax, but
        # their parameterisation-specific minimums are deliberately left to grompp.
        if name in settings:
            cutoff_values[name] = numeric_setting(name)

    if "cutoffscheme" in settings:
        scheme_tokens = settings["cutoffscheme"].split()
        cutoff_scheme = (scheme_tokens[0].lower().replace("_", "-")
                         if scheme_tokens else "<empty>")
        if cutoff_scheme != "verlet":
            problems.append(
                f"cutoff-scheme={cutoff_scheme} is unsupported; use Verlet")

    # Job-local custom *.ff directories are an intentional expert workflow.
    # Apply structural checks, then defer family-specific policy to grompp.
    if family is None:
        if problems:
            raise ValueError(
                f"MDP '{os.path.basename(mdp_file_path)}' has invalid settings for "
                f"custom topology force field '{force_field}': "
                + "; ".join(problems)
            )
        return force_field

    required_cutoff = (
        1.4 if family == "GROMOS" else 1.2 if family == "CHARMM" else 1.0)
    # Under Verlet, GROMOS rlist is an automatically managed neighbour-list
    # distance and its PME/RF real-space electrostatics may use rcoulomb=1.0.
    cutoff_names = (
        ("rvdw",) if family == "GROMOS"
        else ("rlist", "rvdw", "rcoulomb"))
    for name in cutoff_names:
        value = cutoff_values.get(name, 1.0)
        if value is not None and value < required_cutoff:
            problems.append(
                f"{name}={value:g} nm is below the {family} minimum "
                f"of {required_cutoff:g} nm"
            )

    coulomb_tokens = settings.get("coulombtype", "cut-off").split()
    coulomb = coulomb_tokens[0] if coulomb_tokens else "<empty>"
    normalized_coulomb = coulomb.lower().replace("_", "-")
    if family in ("AMBER", "CHARMM", "OPLS"):
        if not normalized_coulomb.startswith("pme"):
            problems.append(
                f"coulombtype={coulomb} is incompatible with {family}; use PME")
    elif family == "GROMOS":
        permitted = normalized_coulomb.startswith("pme") or normalized_coulomb in {
            "reaction-field", "reaction-field-zero", "reactionfield",
            "reactionfieldzero",
        }
        if not permitted:
            problems.append(
                f"coulombtype={coulomb} is incompatible with GROMOS; use "
                "Reaction-Field or PME")

    dispersion_tokens = settings.get("dispcorr", "no").split()
    dispersion = dispersion_tokens[0].lower() if dispersion_tokens else "<empty>"
    if family in ("AMBER", "OPLS"):
        if dispersion not in ("enerpres", "allenerpres"):
            problems.append(
                f"DispCorr={dispersion} does not include energy/pressure correction")
    elif dispersion != "no":
        problems.append(f"DispCorr={dispersion} is incompatible with {family}")

    if family == "CHARMM":
        charmm_rvdw = cutoff_values.get("rvdw", 1.0)
        if charmm_rvdw is not None and not math.isclose(
                charmm_rvdw, 1.2, rel_tol=0.0, abs_tol=1e-6):
            problems.append("rvdw must be exactly 1.2 nm for CHARMM force-switch")
        modifier = settings.get("vdwmodifier", "").split()
        modifier_value = modifier[0].lower().replace("_", "-") if modifier else ""
        if modifier_value != "force-switch":
            problems.append("vdw-modifier must be force-switch for CHARMM")
        switch = numeric_setting("rvdwswitch", default=0.0)
        if switch is not None and not math.isclose(
                switch, 1.0, rel_tol=0.0, abs_tol=1e-6):
            problems.append("rvdw-switch must be 1.0 nm for CHARMM")

    integrator_tokens = settings.get("integrator", "md").split()
    integrator = (integrator_tokens[0].lower().replace("_", "-")
                  if integrator_tokens else "md")
    dynamics_integrators = {"md", "md-vv", "md-vv-avek", "sd", "bd"}
    if (family == "GROMOS" and integrator in dynamics_integrators
            and time_step is not None and time_step > 0.001):
        constraint_tokens = settings.get("constraints", "none").split()
        constraints = constraint_tokens[0] if constraint_tokens else "<empty>"
        normalized_constraints = constraints.lower().replace("_", "-")
        if normalized_constraints not in ("all-bonds", "allbonds"):
            problems.append(
                f"constraints={constraints} is unsafe for GROMOS at "
                f"dt={time_step:g} ps; use all-bonds above 0.001 ps")

    if problems:
        raise ValueError(
            f"MDP '{os.path.basename(mdp_file_path)}' is incompatible with "
            f"{family} topology force field '{force_field}': " + "; ".join(problems)
        )
    return force_field


def _read_moleculetype_names(topology_file_path: str) -> list[str]:
    """Read all literal molecule names declared by ``[ moleculetype ]`` sections."""
    names: list[str] = []
    in_moleculetype = False
    with open(topology_file_path, "r") as topology_file:
        for line in topology_file:
            section_name = _gromacs_section_name(line)
            if section_name is not None:
                in_moleculetype = section_name == "moleculetype"
                continue

            content = line.split(";", 1)[0].strip()
            if in_moleculetype and content and not content.startswith("#"):
                names.append(content.split()[0])
                in_moleculetype = False
    return names


def _read_moleculetype_name(topology_file_path: str) -> str:
    """Read the sole molecule name declared by a ligand ITP."""
    names = _read_moleculetype_names(topology_file_path)
    if len(names) == 1:
        return names[0]
    if len(names) > 1:
        raise ValueError(
            f"Ligand topology '{os.path.basename(topology_file_path)}' declares "
            f"multiple molecule types ({', '.join(names)}); this merge action "
            "supports exactly one ligand molecule type."
        )

    raise ValueError(
        f"Ligand topology '{os.path.basename(topology_file_path)}' does not contain "
        "a molecule name in a [ moleculetype ] section."
    )


def _ligand_posre_file_name(ligand_topology_file_name: str) -> str:
    """Return ACPYPE's restraint filename for one of its ``*_GMX.itp`` files."""
    stem, _ = os.path.splitext(ligand_topology_file_name)
    if stem.lower().endswith("_gmx"):
        stem = stem[:-4]
    return f"posre_{stem}.itp"


def _is_named_include(line: str, file_name: str) -> bool:
    include_name = _gromacs_include_name(line)
    return include_name is not None and os.path.basename(include_name) == file_name


def merge_protein_ligand_topologies(protein_topology_file_path: str,
                                    ligand_topology_file_path: str,
                                    output_topology_file_path: str,
                                    ligand_structure_file_path: str | None = None) -> None:
    """Include a ligand ITP and list its declared molecule once in ``[ molecules ]``."""
    ligand_file_name = os.path.basename(ligand_topology_file_path)
    molecule_name = _read_moleculetype_name(ligand_topology_file_path)
    ligand_structure_file_path = (
        ligand_structure_file_path
        or _existing_acpype_counterpart(
            ligand_topology_file_path, ".itp", ".gro"))
    if ligand_structure_file_path is not None:
        validate_ligand_gro_itp_pair(
            ligand_structure_file_path, ligand_topology_file_path)
    posre_file_name = _ligand_posre_file_name(ligand_file_name)
    posre_path = os.path.join(os.path.dirname(ligand_topology_file_path), posre_file_name)

    with open(protein_topology_file_path, "r") as protein_topology:
        source_lines = protein_topology.readlines()

    if not any(
        (_gromacs_include_name(line) is not None
         and os.path.basename(_gromacs_include_name(line) or "") == "forcefield.itp")
        for line in source_lines
    ):
        raise ValueError(
            f"Protein topology '{os.path.basename(protein_topology_file_path)}' has no "
            "forcefield.itp include; cannot place the ligand atom types safely."
        )

    # Drop the block written on an earlier merge. The explicit markers make the
    # complete operation idempotent while keeping unrelated user content intact.
    lines: list[str] = []
    in_managed_block = False
    previous_managed_molecules: set[str] = set()
    unmanaged_selected_include = False
    for line in source_lines:
        if line.strip() == _LIGAND_BLOCK_BEGIN:
            in_managed_block = True
            if lines and not lines[-1].strip():
                lines.pop()
            continue
        if in_managed_block:
            stripped = line.strip()
            if stripped.startswith(_LIGAND_MOLECULE_NAME_PREFIX):
                previous_name = stripped[len(_LIGAND_MOLECULE_NAME_PREFIX):].strip()
                if previous_name:
                    previous_managed_molecules.add(previous_name)
            else:
                # Topologies produced before the explicit molecule marker can
                # still be upgraded when their managed ITP remains beside them.
                include_name = _gromacs_include_name(line)
                if (include_name is not None
                        and os.path.basename(include_name) == include_name
                        and not include_name.lower().startswith("posre_")):
                    previous_itp = os.path.join(
                        os.path.dirname(protein_topology_file_path), include_name)
                    if os.path.isfile(previous_itp):
                        try:
                            previous_managed_molecules.add(
                                _read_moleculetype_name(previous_itp))
                        except (OSError, UnicodeError, ValueError):
                            pass
            if line.strip() == _LIGAND_BLOCK_END:
                in_managed_block = False
            continue
        if _is_named_include(line, ligand_file_name):
            unmanaged_selected_include = True
        lines.append(line)
    if in_managed_block:
        raise ValueError("Protein topology contains an unterminated managed ligand include block.")
    if unmanaged_selected_include:
        raise ValueError(
            f"Protein topology already includes '{ligand_file_name}' outside a "
            "GROMACS WebUI managed ligand block. Remove or rename that unmanaged "
            "include before merging."
        )

    inline_molecule_types = set(_read_moleculetype_names(
        protein_topology_file_path))
    if molecule_name in inline_molecule_types:
        raise ValueError(
            f"Protein topology already declares molecule type '{molecule_name}' "
            "outside the managed ligand include. Choose a ligand with a unique "
            "molecule type or remove the conflicting declaration."
        )

    # pdb2gmx commonly puts protein molecule types in sibling ITPs. Inspect only
    # plain local includes: following absolute/nested include paths would escape
    # this topology's self-contained job directory and is unnecessary here.
    topology_directory = os.path.dirname(protein_topology_file_path)
    for line in lines:
        include_name = _gromacs_include_name(line)
        if (include_name is None or os.path.basename(include_name) != include_name
                or include_name == ligand_file_name):
            continue
        included_path = os.path.join(topology_directory, include_name)
        if not os.path.isfile(included_path):
            continue
        try:
            included_types = _read_moleculetype_names(included_path)
        except (OSError, UnicodeError):
            continue
        if molecule_name in included_types:
            raise ValueError(
                f"Protein topology include '{include_name}' already declares "
                f"molecule type '{molecule_name}'. The ligand molecule type must "
                "be unique."
            )

    forcefield_index = next(
        index for index, line in enumerate(lines)
        if (_gromacs_include_name(line) is not None
            and os.path.basename(_gromacs_include_name(line) or "") == "forcefield.itp")
    )
    ligand_block = [
        "\n",
        f"{_LIGAND_BLOCK_BEGIN}\n",
        "; Include ligand topology\n",
        f"{_LIGAND_MOLECULE_NAME_PREFIX}{molecule_name}\n",
        f'#include "{ligand_file_name}"\n',
    ]
    if os.path.isfile(posre_path):
        ligand_block.extend([
            "\n",
            "; Include ligand position restraints\n",
            "#ifdef POSRES_LIG\n",
            f'#include "{posre_file_name}"\n',
            "#endif\n",
        ])
    ligand_block.append(f"{_LIGAND_BLOCK_END}\n")
    lines[forcefield_index + 1:forcefield_index + 1] = ligand_block

    # Preserve the position of the first existing entry because molecule order
    # follows coordinate order. Canonicalise its count to one and remove repeats.
    merged_lines: list[str] = []
    in_molecules = False
    molecules_header_index: int | None = None
    ligand_entry_added = False
    legacy_managed_entries = 0
    in_managed_molecule = False
    for line in lines:
        if line.strip() == _LIGAND_MOLECULE_BEGIN:
            in_managed_molecule = True
            continue
        if in_managed_molecule:
            if line.strip() == _LIGAND_MOLECULE_END:
                in_managed_molecule = False
            continue

        section_name = _gromacs_section_name(line)
        if section_name is not None:
            in_molecules = section_name == "molecules"
            if in_molecules and molecules_header_index is None:
                molecules_header_index = len(merged_lines)

        content = line.split(";", 1)[0].strip()
        tokens = content.split()
        if (in_molecules and tokens and tokens[0] == molecule_name
                and not content.startswith("#")
                and molecule_name not in previous_managed_molecules):
            raise ValueError(
                f"The [ molecules ] section already contains unmanaged molecule "
                f"type '{molecule_name}'. Remove that row or use its existing "
                "topology instead of merging another definition."
            )
        if (in_molecules and tokens
                and tokens[0] in previous_managed_molecules
                and not content.startswith("#")):
            legacy_managed_entries += 1
            if legacy_managed_entries > 1:
                raise ValueError(
                    "The [ molecules ] section contains multiple unmarked rows "
                    "for the previously managed ligand; their provenance is "
                    "ambiguous. Remove the duplicates before merging."
                )
            if tokens[0] == molecule_name and not ligand_entry_added:
                merged_lines.extend([
                    f"{_LIGAND_MOLECULE_BEGIN}\n",
                    f"{molecule_name:<18} 1\n",
                    f"{_LIGAND_MOLECULE_END}\n",
                ])
                ligand_entry_added = True
            continue
        merged_lines.append(line)

    if in_managed_molecule:
        raise ValueError("Protein topology contains an unterminated managed ligand molecule block.")

    if molecules_header_index is None:
        if merged_lines and merged_lines[-1].strip():
            merged_lines.append("\n")
        merged_lines.extend([
            "[ molecules ]\n",
            "; Compound        #mols\n",
            f"{_LIGAND_MOLECULE_BEGIN}\n",
            f"{molecule_name:<18} 1\n",
            f"{_LIGAND_MOLECULE_END}\n",
        ])
    elif not ligand_entry_added:
        insert_at = len(merged_lines)
        for index in range(molecules_header_index + 1, len(merged_lines)):
            if _gromacs_section_name(merged_lines[index]) is not None:
                insert_at = index
                break
        while insert_at > molecules_header_index + 1 and not merged_lines[insert_at - 1].strip():
            insert_at -= 1
        merged_lines[insert_at:insert_at] = [
            f"{_LIGAND_MOLECULE_BEGIN}\n",
            f"{molecule_name:<18} 1\n",
            f"{_LIGAND_MOLECULE_END}\n",
        ]

    # Publish only the fully constructed topology so a disk/write failure never
    # leaves a truncated replacement behind.
    atomic_write_text_file(output_topology_file_path, "".join(merged_lines))
WATER_RESNAMES: list[str] = ["SOL", "WAT", "HOH", "H2O", "W", "DOD", "D3O", "TIP3", "TIP3P", "TIP4", "TIP4P", "SPC", "SPCE"]

# Ion resname aliases used by the force fields in play (CHARMM ports, AMBER, GROMACS).
ION_RESNAME_ALIASES: dict[str, str] = {
    "SOD": "NA", "CLA": "CL", "POT": "K", "CAL": "CA", "CES": "CS",
    "LIT": "LI", "IOD": "I", "BAR": "BA", "CAD": "CD",
}

# Jmol/CPK colours, the same palette NGL uses for its element colour scheme.
ELEMENT_COLORS: dict[str, str] = {
    "NA": "#AB5CF2", "CL": "#1FF01F", "K": "#8F40D4", "CA": "#3DFF00", "MG": "#8AFF00",
    "ZN": "#7D80B0", "FE": "#E06633", "CU": "#C88033", "MN": "#9C7AC7", "CO": "#F090A0",
    "NI": "#50D050", "CD": "#FFD98F", "HG": "#B8B8D0", "AG": "#C0C0C0", "AU": "#FFD123",
    "AL": "#BFA6A6", "LI": "#CC80FF", "RB": "#702EB0", "CS": "#57178F", "SR": "#00FF00",
    "BA": "#00C900", "BR": "#A62929", "I": "#940094", "F": "#90E050", "PB": "#575961",
    "CR": "#8A99C7", "BE": "#C2FF00", "GA": "#C28F8F", "IN": "#A67573", "LA": "#70D4FF",
    "CE": "#FFFFC7", "EU": "#61FFC7", "GD": "#45FFC7", "DY": "#1FFFC7", "ER": "#00E675",
    "HO": "#00FF9C", "LU": "#00AB24", "BI": "#9E4FB5", "AS": "#BD80E3", "SE": "#FFA100",
}
# Magenta means "this species was not recognised", so it stands out instead of
# silently blending in with a default colour.
UNKNOWN_SPECIES_COLOR: str = "#FF00FF"

def get_ion_element(resname: str | None) -> str | None:
    """Derive the element symbol from an ion residue name, or None if unrecognised.

    Ion resnames vary by force field: plain (NA, CL, ZN), aliased (SOD, CLA, POT)
    or carrying a charge suffix as in the CHARMM ports (CU2P, FE3P, ZN2P, AG1P).
    Elements are derived from the name rather than guessed from coordinates
    because atom names like 'CU2P' guess badly (carbon, not copper)."""
    name = str(resname or "").strip().upper()
    if not name:
        return None

    if name in ION_RESNAME_ALIASES:
        name = ION_RESNAME_ALIASES[name]
    if name in ELEMENT_COLORS:
        return name

    # Strip a trailing charge marker: CU2P / FE3M / NA+ / CL- / ZN2
    candidate = re.sub(r'(\d+[PM]|[+-]+|\d+)$', '', name)
    if candidate in ION_RESNAME_ALIASES:
        candidate = ION_RESNAME_ALIASES[candidate]
    if candidate in ELEMENT_COLORS:
        return candidate

    return None

def get_structure_species(atoms: mda.Universe | mda.AtomGroup) -> StructureSpecies:
    """Group the atoms into protein / ions / other hetero / water for rendering.

    A resname counts as an ion when its residues hold exactly one atom, which is
    how NA, CL and CU2P are distinguished from a 33-atom LIG without keeping a
    list of ion names (NGL's own 'ion' keyword knows CU but not CU2P)."""
    water_resnames = []
    ions = []
    hetero = []

    protein_atoms = atoms.select_atoms("protein")
    protein_residues = protein_atoms.n_residues
    protein_indices = set(protein_atoms.indices)

    seen = {}
    for residue in atoms.residues:
        resname = str(residue.resname).strip().upper()
        n_atoms_in_residue = len(residue.atoms)
        if resname in WATER_RESNAMES:
            if resname not in water_resnames:
                water_resnames.append(resname)
            continue
        if protein_indices and set(residue.atoms.indices) <= protein_indices:
            continue

        entry = seen.setdefault(resname, {"resname": resname, "count": 0, "atoms_per_residue": n_atoms_in_residue})
        entry["count"] += 1

    for entry in sorted(seen.values(), key=lambda e: (-e["count"], e["resname"])):
        if entry["atoms_per_residue"] == 1:
            element = get_ion_element(entry["resname"])
            ions.append({
                "resname": entry["resname"],
                "count": entry["count"],
                "element": element,
                "color": ELEMENT_COLORS.get(element, UNKNOWN_SPECIES_COLOR),
                "recognized": element is not None,
            })
        else:
            hetero.append(entry)

    return {"protein_residues": protein_residues, "ions": ions, "hetero": hetero, "water": water_resnames}

def get_species_legend(species: StructureSpecies) -> str:
    """One-line summary of the species on show, flagging unrecognised ones."""
    parts = []
    if species["protein_residues"]:
        parts.append(f"protein {species['protein_residues']} res")
    for entry in species["hetero"]:
        resname = html.escape(str(entry["resname"]), quote=True)
        parts.append(
            f"{resname} {entry['count']} ({entry['atoms_per_residue']} atoms)")
    for ion in species["ions"]:
        resname = html.escape(str(ion["resname"]), quote=True)
        parts.append(
            f"{resname} {ion['count']}" if ion["recognized"]
            else f"{resname} {ion['count']} (unrecognised, magenta)")
    for resname in species["water"]:
        parts.append(f"{html.escape(str(resname), quote=True)} (water)")

    return "Species: " + ", ".join(parts) + "." if parts else ""

def get_species_representations_js(species: StructureSpecies) -> str:
    """NGL.js representation calls for every species present in the structure."""
    lines = []
    n_protein_residues = species["protein_residues"]

    if n_protein_residues:
        # Cartoon needs a backbone long enough to trace; on a short peptide it
        # draws nothing and the viewport looks empty, so use licorice there.
        if n_protein_residues < 20:
            lines.append('comp.addRepresentation("licorice", { sele: "protein", colorScheme: "element" });')
        else:
            lines.append('comp.addRepresentation("cartoon", { sele: "protein", colorScheme: "sstruc" });')

    for entry in species["hetero"]:
        selection = json.dumps(f'[{entry["resname"]}]')
        lines.append(
            f'comp.addRepresentation("ball+stick", {{ sele: {selection} }});')

    for ion in species["ions"]:
        # Explicit colour and a fixed radius: a mis-guessed element must not be
        # able to shrink or grey out an ion sphere.
        selection = json.dumps(f'[{ion["resname"]}]')
        color = json.dumps(str(ion["color"]))
        lines.append(
            f'comp.addRepresentation("spacefill", {{ sele: {selection}, '
            f'color: {color}, radiusType: "size", radiusSize: 1.0 }});')

    if species["ions"]:
        ion_selection = " or ".join(f'[{ion["resname"]}]' for ion in species["ions"])
        lines.append(
            f'comp.addRepresentation("label", {{ sele: {json.dumps(ion_selection)}, '
            'labelType: "resname", color: "#222222", scale: 1.5, '
            'showBackground: true, backgroundColor: "white", '
            'backgroundOpacity: 0.5 });')

    if species["water"]:
        lines.append('comp.addRepresentation("line", { sele: "water", opacity: 0.3 });')

    return "\n  ".join(lines)

def add_species_representations_to_nglview(view: nglview.NGLWidget, species: StructureSpecies) -> None:
    """Same species handling as the trajectory viewer, applied to an nglview widget."""
    view.clear()

    n_protein_residues = species["protein_residues"]
    if n_protein_residues:
        if n_protein_residues < 20:
            view.add_representation("licorice", selection="protein")
        else:
            view.add_cartoon("protein", color="sstruc")

    for entry in species["hetero"]:
        view.add_ball_and_stick(f'[{entry["resname"]}]')

    for ion in species["ions"]:
        view.add_representation("spacefill", selection=f'[{ion["resname"]}]', color=ion["color"], radiusType="size", radiusSize=1.0)

    if species["ions"]:
        ion_selection = " or ".join(f'[{ion["resname"]}]' for ion in species["ions"])
        view.add_representation("label", selection=ion_selection, labelType="resname", color="#222222", scale=1.5,
                                showBackground=True, backgroundColor="white", backgroundOpacity=0.5)

    if species["water"]:
        view.add_representation("line", selection="water", opacity=0.3)

def prepare_structure_viewer_file(structure_file_path: str,
                                  static_output_path: str) -> tuple[str, StructureSpecies]:
    """Return (path NGL should load, species present) for the structure viewer.

    Non-PDB inputs are converted with MDAnalysis rather than ParmEd: ParmEd clips
    residue names to the 3-column PDB field (CU2P -> CU2), which would stop the
    per-species selections from matching. MDAnalysis writes 4 characters into
    columns 18-21, exactly what NGL's PDB parser reads."""
    universe = mda.Universe(structure_file_path)

    try:
        try:
            universe.guess_TopologyAttrs(context="default", to_guess=["elements"])
        except Exception:
            pass

        if structure_file_path.lower().endswith(".pdb"):
            display_path = structure_file_path
        else:
            universe.atoms.write(static_output_path)
            display_path = static_output_path

        return display_path, get_structure_species(universe)
    finally:
        universe.trajectory.close()

TRAJECTORY_VIEWER_SELECTIONS: dict[str, str] = {
    "Protein": "protein",
    "Protein + Ligand + Ions": "not resname " + " ".join(WATER_RESNAMES),
    "All Atoms": "all",
}
MAX_TRAJECTORY_VIEWER_COORDINATES = 5_000_000

def write_trajectory_viewer_files(structure_file_path: str, trajectory_file_path: str, selection: str,
                                  max_frames: int, static_basename: str) -> TrajectoryViewerInfo:
    """Write a reduced structure/trajectory pair into ./static for the NGL viewer.

    Production trajectories here run to several GB of solvated system, which no
    browser can hold in memory (NGL keeps every frame as float32), so the frames
    are subsetted and strided before they are handed over."""
    if isinstance(max_frames, (bool, np.bool_)):
        raise ValueError("Max Frames must be an integer from 1 to 1000.")
    try:
        numeric_max_frames = float(max_frames)
    except (TypeError, ValueError):
        raise ValueError("Max Frames must be an integer from 1 to 1000.") from None
    if (not math.isfinite(numeric_max_frames)
            or not numeric_max_frames.is_integer()
            or not 1 <= numeric_max_frames <= 1000):
        raise ValueError("Max Frames must be an integer from 1 to 1000.")
    frame_limit = int(numeric_max_frames)

    selection_string = TRAJECTORY_VIEWER_SELECTIONS.get(selection, selection)

    structure_name = os.path.basename(structure_file_path)
    trajectory_name = os.path.basename(trajectory_file_path)

    try:
        universe = mda.Universe(structure_file_path, trajectory_file_path)
    except Exception as exc:
        # Mismatched atom counts are the usual cause, but an empty or truncated
        # trajectory lands here too, so do not assert one over the other.
        raise Exception(
            f"Could not read '{structure_name}' together with '{trajectory_name}'. Check that both "
            f"come from the same run (so the atom counts match) and that the trajectory is complete.\n{exc}"
        ) from exc

    if len(universe.trajectory) == 0:
        universe.trajectory.close()
        raise Exception(f"'{trajectory_name}' contains no frames.")

    atoms = universe.select_atoms(selection_string)
    if atoms.n_atoms == 0:
        universe.trajectory.close()
        raise Exception(f"Selection '{selection}' matched no atoms in {structure_name}.")

    # GRO files carry no element column, so fill it in before writing the PDB;
    # NGL uses elements for bond detection, colours and radii.
    try:
        universe.guess_TopologyAttrs(context="default", to_guess=["elements"])
    except Exception:
        pass

    total_frames = len(universe.trajectory)
    if atoms.n_atoms > MAX_TRAJECTORY_VIEWER_COORDINATES:
        universe.trajectory.close()
        raise ValueError(
            f"Selection '{selection}' contains {atoms.n_atoms:,} atoms, which is "
            "too large for the browser viewer. Choose a narrower selection.")
    frame_limit = min(
        frame_limit,
        max(1, MAX_TRAJECTORY_VIEWER_COORDINATES // atoms.n_atoms),
    )
    stride = max(1, math.ceil(total_frames / frame_limit))

    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    structure_output_path = str(STATIC_ROOT / (static_basename + ".pdb"))
    trajectory_output_path = str(STATIC_ROOT / (static_basename + ".xtc"))

    completed = False
    try:
        # Structure and trajectory are written from the same selection so their
        # atom counts always agree, which NGL requires to apply frames to it.
        universe.trajectory[0]
        atoms.write(structure_output_path)

        written_frames = 0
        with mda.Writer(trajectory_output_path, atoms.n_atoms) as writer:
            for _ in universe.trajectory[::stride]:
                writer.write(atoms)
                written_frames += 1

        species = get_structure_species(atoms)
        completed = True
    finally:
        # Release multi-GB trajectory handles on both success and write errors.
        universe.trajectory.close()
        if not completed:
            for partial_path in (structure_output_path, trajectory_output_path):
                try:
                    os.unlink(partial_path)
                except FileNotFoundError:
                    pass

    return {
        "frames": written_frames,
        "stride": stride,
        "total_frames": total_frames,
        "n_atoms": atoms.n_atoms,
        "n_residues": atoms.n_residues,
        "species": species,
    }

TRAJECTORY_VIEWER_HTML_TEMPLATE: str = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MD Trajectory Viewer</title>
<script src="https://unpkg.com/ngl@2.4.0/dist/ngl.js"></script>
<style>
  html, body { margin: 0; padding: 0; font-family: sans-serif; background: #ffffff; }
  #viewport { width: 100%; height: 700px; }
  #controls { display: flex; align-items: center; gap: 8px; padding: 6px; }
  #frame { flex: 1; }
  #counter { font-size: 12px; color: #333; min-width: 80px; text-align: right; }
  #status { font-size: 12px; color: #555; padding: 0 6px 6px 6px; }
</style>
</head>
<body>
<div id="viewport"></div>
<div id="controls">
  <button id="toggle" type="button">Pause</button>
  <input id="frame" type="range" min="0" max="__MAX_FRAME__" value="0" step="1">
  <span id="counter">-</span>
</div>
<div id="status">Loading trajectory...</div>
<script>
var TS = "__TIMESTAMP__";
var BASE = "__BASENAME__";

var statusEl = document.getElementById("status");
var toggleEl = document.getElementById("toggle");
var frameEl = document.getElementById("frame");
var counterEl = document.getElementById("counter");

var stage = new NGL.Stage("viewport", { backgroundColor: "white" });
window.addEventListener("resize", function () { stage.handleResize(); });

stage.loadFile("/static/" + BASE + ".pdb?ts=" + TS).then(function (comp) {
  __SPECIES_REPRESENTATIONS__
  comp.autoView();

  return NGL.autoLoad("/static/" + BASE + ".xtc?ts=" + TS).then(function (frames) {
    var traj = comp.addTrajectory(frames, {}).trajectory;
    var count = traj.frameCount;
    frameEl.max = Math.max(0, count - 1);

    var player = new NGL.TrajectoryPlayer(traj, {
      step: 1, timeout: 80, mode: "loop", direction: "forward", interpolateType: ""
    });
    var playing = true;
    player.play();

    toggleEl.addEventListener("click", function () {
      if (playing) { player.pause(); toggleEl.textContent = "Play"; }
      else { player.play(); toggleEl.textContent = "Pause"; }
      playing = !playing;
    });

    frameEl.addEventListener("input", function () {
      if (playing) { player.pause(); playing = false; toggleEl.textContent = "Play"; }
      traj.setFrame(parseInt(frameEl.value, 10));
    });

    setInterval(function () {
      var current = traj.currentFrame;
      if (current < 0) { return; }
      if (playing) { frameEl.value = current; }
      counterEl.textContent = (current + 1) + " / " + count;
    }, 100);

    statusEl.textContent = count + " frames animating.";
  });
}).catch(function (err) {
  statusEl.textContent = "Error: " + err;
});
</script>
</body>
</html>
"""

def get_trajectory_viewer_html(static_basename: str, timestamp: int, n_frames: int,
                               species: StructureSpecies) -> str:
    """Build the self-contained NGL page that animates the reduced trajectory."""
    representations = get_species_representations_js(species)

    return (TRAJECTORY_VIEWER_HTML_TEMPLATE
            .replace("__BASENAME__", static_basename)
            .replace("__TIMESTAMP__", str(timestamp))
            .replace("__MAX_FRAME__", str(max(0, n_frames - 1)))
            .replace("__SPECIES_REPRESENTATIONS__", representations))
