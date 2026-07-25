"""Shared helpers for the GROMACS WebUI: MDP generation, GROMACS process
handling, topology merging and structure/trajectory viewer support."""

from __future__ import annotations

import math
import os
import re
import subprocess
import threading
from collections.abc import Sequence
from typing import Any, TypedDict

import MDAnalysis as mda
import nglview
import torch
from nnpot_models import (
    GmxAIMNet2Model,
    GmxANI1xModel,
    GmxANI2xEMLEModel,
    GmxANI2xModel,
    GmxMACEModel,
)
from e3nn.util.jit import script


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
    os.makedirs("./models", exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", os.path.abspath("./models/warp-cache"))
    os.environ.setdefault("AIMNET_CACHE_DIR", os.path.abspath("./models/aimnet-cache"))
    modelfile_path = os.path.join("./models", f"{model_name}.pt")
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

def get_gpu_mdrun_options(use_gpu: bool, mpi_rank: int) -> list[str]:
    """mdrun GPU offload flags that are safe for minimisation and restrained
    equilibration: nonbonded and PME carry almost all of the cost, while
    -bonded gpu and -update gpu are refused outright when position restraints
    are present, and GPU PME needs a single rank."""
    if not use_gpu:
        return []

    options = ["-nb", "gpu"]
    if int(mpi_rank) == 1:
        options.extend(["-pme", "gpu"])

    return options

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
