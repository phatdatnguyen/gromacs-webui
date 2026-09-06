"""Shared helpers for the GROMACS WebUI: MDP generation, GROMACS process
handling, topology merging and structure/trajectory viewer support."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Sequence
from typing import Any, TypedDict

import MDAnalysis as mda
import nglview
import numpy as np
import pandas as pd

# Machine learning potentials are optional. torch, e3nn and the model packages are
# large, so they are neither imported nor required at start-up: availability is
# checked by looking for the modules, and they are imported only when a model is
# actually built. Everything else in the application works without them.
NNPOT_REQUIRED_PACKAGES: tuple[str, ...] = ("torch", "e3nn")


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


class XvgData(TypedDict):
    """A parsed GROMACS .xvg file: the numbers plus the labels it carries itself."""

    frame: pd.DataFrame
    title: str
    xlabel: str
    ylabel: str



def get_missing_nnpot_packages() -> list[str]:
    """Which machine learning potential packages are absent from this environment."""
    missing = []
    for name in NNPOT_REQUIRED_PACKAGES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:
            # A broken or shadowed installation is as good as a missing one here.
            missing.append(name)

    return missing


def get_nnpot_unavailable_reason() -> str | None:
    """A message naming what to install, or None when potentials can be used."""
    missing = get_missing_nnpot_packages()
    if not missing:
        return None

    return (f"Machine learning potentials are disabled: {', '.join(missing)} "
            f"not installed. See the Readme for the optional install steps.")


GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE: str = "GMX_MMPBSA_EXECUTABLE"
# The environment the Readme tells you to build, beside the application's own.
GMX_MMPBSA_ENVIRONMENT_PATH: str = "./gmx-mmpbsa-env"

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
        return configured if _is_executable(configured) else None

    local = os.path.abspath(os.path.join(GMX_MMPBSA_ENVIRONMENT_PATH, "bin", "gmx_MMPBSA"))
    if _is_executable(local):
        return local

    return shutil.which("gmx_MMPBSA")


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


def is_nnpot_available() -> bool:
    """Whether the optional machine learning potential support can be used."""
    return not get_missing_nnpot_packages()


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
        return f"{model_name}|torchani|pyaev|adaptive|extensions-disabled"
    if model_name == "ani2x-emle":
        return f"{model_name}|emle|empty-mm-environment|energy-only-pyaev-v2"
    if model_name.startswith("mace-"):
        return f"{model_name}|mace|internal-neighbors-singular-cell-v4"
    if model_name == "aimnet2":
        return f"{model_name}|aimnet|traced-positions-numbers-box-pbc-device-float64-v5"
    return model_name

def is_cached_nnpot_model_usable(model_name: str, modelfile_path: str) -> bool:
    """Report whether the cached model matches this build, moving it aside if not."""
    import torch

    extra_files = {"nnpot_model_config": ""}
    try:
        torch.jit.load(modelfile_path, map_location="cpu", _extra_files=extra_files)
        cached_config = extra_files["nnpot_model_config"]
        if isinstance(cached_config, bytes):
            cached_config = cached_config.decode()
        if cached_config != get_expected_nnpot_model_config(model_name):
            backup_path = modelfile_path + ".invalid"
            os.replace(modelfile_path, backup_path)
            print(f"Moved outdated cached NNPot model to {backup_path}.")
            return False
        return True
    except RuntimeError as exc:
        if get_nnpot_model_load_error_message(exc) is not None:
            backup_path = modelfile_path + ".invalid"
            os.replace(modelfile_path, backup_path)
            print(f"Moved unusable cached NNPot model to {backup_path}.")
            return False
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
    cell = torch.eye(3, dtype=torch.float32, device=device)
    pbc = torch.tensor([True, True, True], device=device)
    return torch.jit.trace(
        model,
        (positions, atomic_numbers, cell, pbc),
        strict=False,
        check_trace=False,
    )

def download_nnpot_model(model_name: str) -> str:
    """Build or reuse the wrapped neural-network potential and return its file path."""
    reason = get_nnpot_unavailable_reason()
    if reason is not None:
        raise RuntimeError(reason)

    import torch
    from e3nn.util.jit import script
    from nnpot_models import (
        GmxAIMNet2Model,
        GmxANI1xModel,
        GmxANI2xEMLEModel,
        GmxANI2xModel,
        GmxMACEModel,
    )

    os.makedirs("./models", exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", os.path.abspath("./models/warp-cache"))
    os.environ.setdefault("AIMNET_CACHE_DIR", os.path.abspath("./models/aimnet-cache"))
    # Absolute: this path is written into the MDP as nnpot-modelfile and resolved
    # by mdrun, which runs from the job directory rather than the repository root.
    modelfile_path = os.path.abspath(os.path.join("./models", f"{model_name}.pt"))
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
        scripted_model.save(modelfile_path, _extra_files=extensions)
    elif is_emle_model:
        torch.jit.script(model).save(modelfile_path, _extra_files=extensions)
    elif not "mace" in model_name:
        torch.jit.script(model).save(modelfile_path, _extra_files=extensions)
    else:
        # for MACE, we need to use the e3nn scipting function
        scripted_model = script(model)
        scripted_model.save(modelfile_path, _extra_files=extensions)
    print(f"Saved wrapped model to {modelfile_path}.")
    
    return modelfile_path

def run_checked_command(cmd: Sequence[str], cwd: str | None = None, stdin_input: str | None = None,
                        error_lines: int = 25) -> subprocess.CompletedProcess[str]:
    """Run a command to completion, raising an Exception that carries its stderr.

    GROMACS writes its diagnostics ("Fatal error", missing atoms, mismatched
    coordinate counts) to stderr. Without capturing it, a failure surfaces in the
    UI as nothing more than 'returned non-zero exit status 1'."""
    process = subprocess.run(cmd, cwd=cwd, input=stdin_input, text=True, capture_output=True)

    if process.returncode != 0:
        output = (process.stderr or "") + "\n" + (process.stdout or "")
        lines = [line for line in output.splitlines() if line.strip()]

        # GROMACS prints a version banner before the diagnostic, so start the
        # message at the error block itself and fall back to the tail otherwise.
        marker_index = None
        for index, line in enumerate(lines):
            if line.lstrip().startswith(("Fatal error", "Error in user input", "Inconsistency in user input")):
                marker_index = index

        if marker_index is None:
            detail_lines = lines[-error_lines:]
        else:
            detail_lines = [line for line in lines[marker_index:]
                            if not line.startswith("---")
                            and "troubleshooting" not in line
                            and "manual.gromacs.org" not in line][:error_lines]

        detail = "\n".join(detail_lines) if detail_lines else "no output captured"
        raise Exception(f"{os.path.basename(cmd[0])} {cmd[1] if len(cmd) > 1 else ''} failed "
                        f"(exit status {process.returncode}):\n{detail}")

    return process

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
    probe = subprocess.Popen(list(cmd), cwd=cwd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = probe.communicate(input="\n" * 8)

    # trjconv and the analysis tools print the menu on stderr, but not all of them
    # agree on that, so search both streams.
    return parse_gmx_groups(stderr + stdout)

def stop_process_gracefully(proc: subprocess.Popen[str] | None, timeout: float = 15) -> None:
    """Ask a run to stop, only killing it if it ignores the request.

    mdrun handles SIGTERM by finishing the current step, writing a checkpoint and
    a confout structure; SIGKILL would discard everything since the last
    checkpoint was written."""
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Reap it, otherwise the caller still sees a live process for a moment
        # and the child lingers as a zombie.
        proc.wait()

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
    """dict subclass for gr.State that creates a fresh lock on deep copy."""
    def __init__(self) -> None:
        super().__init__({"proc": None, "running": False, "lock": threading.Lock()})

    def __deepcopy__(self, memo: dict[int, Any]) -> ProcessStateDict:
        return ProcessStateDict()

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

def _remove_pdb2gmx_probe_files(working_directory_path: str) -> None:
    """Delete the throwaway outputs, and GROMACS backups, left by a probe run."""
    try:
        names = os.listdir(working_directory_path)
    except OSError:
        return

    for name in names:
        # pdb2gmx also writes per-chain include files and #backup# copies derived
        # from the probe output names.
        if name.startswith(PROBE_PDB2GMX_PREFIX) or name.startswith("#" + PROBE_PDB2GMX_PREFIX):
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
    probe_cmd = list(pdb2gmx_cmd)
    probe_cmd[probe_cmd.index("-o") + 1] = PROBE_PDB2GMX_PREFIX + ".gro"
    probe_cmd[probe_cmd.index("-p") + 1] = PROBE_PDB2GMX_PREFIX + ".top"
    probe_itp = PROBE_PDB2GMX_PREFIX + ".itp"
    if "-i" in probe_cmd:
        probe_cmd[probe_cmd.index("-i") + 1] = probe_itp
    else:
        probe_cmd.extend(["-i", probe_itp])

    try:
        probe = subprocess.Popen(probe_cmd, cwd=working_directory_path, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Answer every prompt up front: pdb2gmx's menu goes to a block-buffered
        # stdout pipe, so reading it before replying would deadlock.
        stdout_probe, stderr_probe = probe.communicate(input="0\n" * 512)
    finally:
        _remove_pdb2gmx_probe_files(working_directory_path)

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

def is_charmm_force_field(force_field: str | None) -> bool:
    """Report whether a force field name belongs to the CHARMM family."""
    return str(force_field or "").strip().lower().startswith("charmm")

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

    return """; Neighbor searching and cutoffs
cutoff-scheme   = Verlet
rlist           = 1.0
rvdw            = 1.0
rcoulomb        = 1.0
coulombtype     = PME"""

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

def get_default_nvt_equilibration_mdp_file_content(time_scale_ps: float = 500, time_step_ps: float = 0.002,
                                                   temperature: float = 300, with_ligand: bool = False,
                                                   force_field: str | None = None) -> str:
    """MDP for restrained NVT equilibration with freshly generated velocities."""
    return f"""
; Restrain the solute while the solvent relaxes around it
define      = -DPOSRES

integrator  = md
dt          = {time_step_ps}
nsteps      = {int(time_scale_ps / time_step_ps)}
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
constraints = h-bonds

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
    return f"""
; Keep the solute restrained through the density equilibration as well
define          = -DPOSRES

integrator      = md
dt              = {time_step_ps}
nsteps          = {int(time_scale_ps / time_step_ps)}

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

; Constraints
constraints     = h-bonds
constraint_algorithm = lincs

{get_cutoff_mdp_section(force_field)}
"""

def get_default_prod_md_mdp_file_content(time_scale_ps: float = 1000, time_step_ps: float = 0.002,
                                         temperature: float = 300, pressure: float = 1.0,
                                         mdp_type: str = "Initial", random_seed: int = 0,
                                         with_ligand: bool = False, nnpot_active: bool = False,
                                         nnpot_modelfile_path: str | None = "models/ani2x.pt",
                                         nnpot_input_group: str = "Protein",
                                         nnpot_model_name: str = "ani2x",
                                         force_field: str | None = None) -> str:
    """MDP for unrestrained production MD, optionally driven by a neural potential."""
    content = f"""
integrator      = md
dt              = {time_step_ps}
nsteps          = {int(time_scale_ps / time_step_ps)}

; Output
nstxout         = 5000
nstvout         = 5000
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
constraints     = h-bonds
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
    if mdp_type=="Initial":
        content = content + f"""
; Continuation
continuation    = no
gen_vel         = yes
gen_temp        = {temperature}
gen_seed        = {random_seed}
"""
    else: # mdp_type=="Continuation"
        content = content + f"""
; Continuation
continuation    = yes
"""

    if nnpot_active:
        content = content + "\n; Neural network potential (machine learning interatomic potential)\n"
        content = content + "nnpot-active          = true\n"
        content = content + f"nnpot-modelfile       = {nnpot_modelfile_path}\n"
        content = content + f"nnpot-input-group     = {nnpot_input_group}\n"
        content = content + """nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = box
nnpot-model-input4    = pbc"""

    return content

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
                                          protein_force_field: str = "leaprc.protein.ff14SB",
                                          ligand_force_field: str = "leaprc.gaff2") -> str:
    """The &general/&gb/&pb/&decomp namelists gmx_MMPBSA reads from its -i file.

    endframe = 0 means "to the end of the trajectory" here; gmx_MMPBSA wants a
    real frame number, so it is only written when the caller gives one.

    The &decomp namelist is what makes the per-residue contributions appear;
    without it gmx_MMPBSA reports only the total binding energy.
    """
    if not use_gb and not use_pb:
        raise ValueError("Select at least one of Generalised Born or Poisson-Boltzmann.")

    content = ("Input file generated by GROMACS WebUI\n"
               "&general\n"
               f"  startframe        = {int(start_frame)},\n")
    if int(end_frame) > 0:
        content += f"  endframe          = {int(end_frame)},\n"
    content += (f"  interval          = {int(interval)},\n"
                f"  temperature       = {float(temperature)},\n"
                f"  forcefields       = \"{protein_force_field}\", \"{ligand_force_field}\",\n"
                "  sys_name          = \"Protein-ligand complex\",\n"
                "  verbose           = 2,\n"
                "/\n")

    if use_gb:
        content += ("&gb\n"
                    f"  igb               = {int(gb_model)},\n"
                    f"  saltcon           = {float(salt_concentration)},\n"
                    "/\n")
    if use_pb:
        content += ("&pb\n"
                    f"  istrng            = {float(salt_concentration)},\n"
                    "  inp               = 2,\n"
                    "  radiopt           = 0,\n"
                    "/\n")

    if use_decomposition:
        # dec_verbose = 3 prints the per-residue breakdown for the complex, the
        # receptor, the ligand and the delta; the delta is the one worth reading,
        # and the others make the difference checkable.
        content += ("&decomp\n"
                    f"  idecomp           = {int(decomposition_scheme)},\n"
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

        heading = stripped.rstrip(":").upper()
        if heading in MMPBSA_METHOD_NAMES:
            method = MMPBSA_METHOD_NAMES[heading]
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

    frame = _read_mmpbsa_csv_block(lines, _find_mmpbsa_section(lines, "Delta Energy Terms"))
    if frame.empty:
        raise ValueError(f"{os.path.basename(csv_file_path)} holds no per-frame energies.")

    return frame

def parse_mmpbsa_decomposition(csv_file_path: str) -> pd.DataFrame:
    """Per-residue contributions to the binding energy, averaged over frames.

    Read from the DELTAS section of gmx_MMPBSA's -deo file, which reports every
    printed residue for every frame; the mean is the contribution and the spread
    across frames says how steady it is.
    """
    with open(csv_file_path) as handle:
        lines = handle.readlines()

    per_frame = _read_mmpbsa_csv_block(
        lines, _find_mmpbsa_section(lines, "DELTAS:", "Total Decomposition Contribution"))
    if per_frame.empty:
        raise ValueError(f"{os.path.basename(csv_file_path)} holds no decomposition data.")

    value_columns = [c for c in per_frame.columns
                     if c not in ("Frame #", "Residue") and per_frame[c].dtype.kind in "if"]
    grouped = per_frame.groupby("Residue", sort=False)
    frame = grouped[value_columns].mean().reset_index()
    frame["TOTAL SD"] = grouped["TOTAL"].std().to_numpy() if "TOTAL" in value_columns else float("nan")

    return frame.sort_values("TOTAL").reset_index(drop=True) if "TOTAL" in frame else frame

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
    total = len(universe.trajectory)
    times: list[float] = []
    for number in range(count):
        index = (start_frame - 1) + number * interval
        if index >= total:
            break
        times.append(universe.trajectory[index].time / 1000)
    universe.trajectory.close()

    return times

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
    counts, x_edges, y_edges = np.histogram2d(np.asarray(x_values, dtype=float),
                                              np.asarray(y_values, dtype=float),
                                              bins=int(bin_count))
    if counts.sum() == 0:
        raise ValueError("The projection contains no points to build a landscape from.")

    probability = counts / counts.sum()
    free_energy = np.full(probability.shape, np.nan)
    populated = probability > 0
    free_energy[populated] = (-BOLTZMANN_CONSTANT_KJ_PER_MOL_K * float(temperature)
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

    indices = frame.iloc[:count, 0].to_numpy()
    eigenvalues = frame.iloc[:count, 1].to_numpy()
    cumulative = np.cumsum(eigenvalues) / frame.iloc[:, 1].to_numpy().sum() * 100

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

    title = lines[0].strip()
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms].strip()

    return title, natoms, atoms, box

def merge_protein_ligand_structures(protein_structure_file_path: str, ligand_structure_file_path: str,
                                    output_structure_file_path: str) -> None:
    """Concatenate protein and ligand coordinates, keeping the protein box."""
    # Read input files
    _, p_n, p_atoms, p_box = read_gromacs_structure_file(protein_structure_file_path)
    _, l_n, l_atoms, _ = read_gromacs_structure_file(ligand_structure_file_path)

    # Combination
    total_atoms = p_n + l_n

    with open(output_structure_file_path, "w") as out:
        out.write("Protein + ligand complex\n")
        out.write(f"{total_atoms}\n")

        for line in p_atoms:
            out.write(line)

        for line in l_atoms:
            out.write(line)

        # keep protein box (ligand box is meaningless alone)
        out.write(p_box + "\n")

def merge_protein_ligand_topologies(protein_topology_file_path: str, ligand_topology_file_path: str,
                                    output_topology_file_path: str) -> None:
    """Include the ligand topology in the protein topology and list it under molecules."""
    ligand_topology_file_name = os.path.basename(ligand_topology_file_path)

    with open(protein_topology_file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    ligand_include_added = False
    molecules_section_found = False
    ligand_in_molecules = False

    for _, line in enumerate(lines):
        new_lines.append(line)

        # 1Insert ligand include after forcefield.itp
        if (
            not ligand_include_added
            and line.strip().startswith('#include')
            and 'forcefield.itp' in line
        ):
            new_lines.append('\n; Include ligand topology\n')
            new_lines.append(f'#include "{ligand_topology_file_name}"\n')
            ligand_include_added = True

        # Detect [ molecules ] section
        if line.strip().lower() == "[ molecules ]":
            molecules_section_found = True

        # Check if ligand already listed
        if molecules_section_found:
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == "ligand":
                ligand_in_molecules = True

    # 2️Append ligand to [ molecules ]
    if not molecules_section_found:
        new_lines.append('\n[ molecules ]\n')
        new_lines.append('; Compound        #mols\n')

    if not ligand_in_molecules:
        new_lines.append('ligand            1\n')

    # Write merged topology
    with open(output_topology_file_path, "w") as f:
        f.writelines(new_lines)

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
        parts.append(f"{entry['resname']} {entry['count']} ({entry['atoms_per_residue']} atoms)")
    for ion in species["ions"]:
        parts.append(f"{ion['resname']} {ion['count']}" if ion["recognized"] else f"{ion['resname']} {ion['count']} (unrecognised, magenta)")
    for resname in species["water"]:
        parts.append(f"{resname} (water)")

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
        lines.append(f'comp.addRepresentation("ball+stick", {{ sele: "[{entry["resname"]}]" }});')

    for ion in species["ions"]:
        # Explicit colour and a fixed radius: a mis-guessed element must not be
        # able to shrink or grey out an ion sphere.
        lines.append(f'comp.addRepresentation("spacefill", {{ sele: "[{ion["resname"]}]", color: "{ion["color"]}", radiusType: "size", radiusSize: 1.0 }});')

    if species["ions"]:
        ion_selection = " or ".join(f'[{ion["resname"]}]' for ion in species["ions"])
        lines.append(f'comp.addRepresentation("label", {{ sele: "{ion_selection}", labelType: "resname", color: "#222222", scale: 1.5, showBackground: true, backgroundColor: "white", backgroundOpacity: 0.5 }});')

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
        universe.guess_TopologyAttrs(context="default", to_guess=["elements"])
    except Exception:
        pass

    if structure_file_path.endswith(".pdb"):
        display_path = structure_file_path
    else:
        universe.atoms.write(static_output_path)
        display_path = static_output_path

    return display_path, get_structure_species(universe)

TRAJECTORY_VIEWER_SELECTIONS: dict[str, str] = {
    "Protein": "protein",
    "Protein + Ligand + Ions": "not resname " + " ".join(WATER_RESNAMES),
    "All Atoms": "all",
}

def write_trajectory_viewer_files(structure_file_path: str, trajectory_file_path: str, selection: str,
                                  max_frames: int, static_basename: str) -> TrajectoryViewerInfo:
    """Write a reduced structure/trajectory pair into ./static for the NGL viewer.

    Production trajectories here run to several GB of solvated system, which no
    browser can hold in memory (NGL keeps every frame as float32), so the frames
    are subsetted and strided before they are handed over."""
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
        raise Exception(f"'{trajectory_name}' contains no frames.")

    atoms = universe.select_atoms(selection_string)
    if atoms.n_atoms == 0:
        raise Exception(f"Selection '{selection}' matched no atoms in {structure_name}.")

    # GRO files carry no element column, so fill it in before writing the PDB;
    # NGL uses elements for bond detection, colours and radii.
    try:
        universe.guess_TopologyAttrs(context="default", to_guess=["elements"])
    except Exception:
        pass

    total_frames = len(universe.trajectory)
    stride = max(1, math.ceil(total_frames / max(1, int(max_frames))))

    structure_output_path = os.path.join("./static", static_basename + ".pdb")
    trajectory_output_path = os.path.join("./static", static_basename + ".xtc")

    # Structure and trajectory are written from the same selection so their atom
    # counts always agree, which NGL requires to apply frames to the structure.
    universe.trajectory[0]
    atoms.write(structure_output_path)

    written_frames = 0
    with mda.Writer(trajectory_output_path, atoms.n_atoms) as writer:
        for _ in universe.trajectory[::stride]:
            writer.write(atoms)
            written_frames += 1

    species = get_structure_species(atoms)
    # Release the reader's file handle now rather than at the next collection: a
    # multi-GB trajectory would otherwise stay open for the life of the server.
    universe.trajectory.close()

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
