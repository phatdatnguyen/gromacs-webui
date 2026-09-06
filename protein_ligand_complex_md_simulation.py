import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import html
import math
import numbers
import re
import time
import threading
import psutil
import shutil
import filecmp
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr
import nglview
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms
import matplotlib.pyplot as plt
from utils import *
from collections.abc import Sequence
from typing import Any

# What gr.update() hands back to Gradio.
GradioUpdate = dict[str, Any]
from path_security import (
    DATA_ROOT,
    STATIC_ROOT,
    atomic_replace_editable_text_file,
    cleanup_stale_static_assets,
    read_editable_text_file,
    remove_static_asset_bundle,
    secure_module_callbacks,
    static_asset_basename,
    validate_file_name,
)


MAX_STANDARD_TIME_STEP_PS = 0.002
MAX_DISTANCE_ARRAY_PAIRS = 1_000_000
_FORCE_FIELD_INCLUDE_RE = re.compile(
    r'^\s*#include\s+["<]([^">]+)\.ff/forcefield\.itp[">]',
    re.IGNORECASE | re.MULTILINE,
)

_THREE_SITE_WATER_MODELS = {"SPC", "SPCE", "TIP3P", "OPC3", "TIPS3P"}
_FOUR_SITE_WATER_MODELS = {"TIP4P", "TIP4PEW", "OPC"}
_KNOWN_WATER_MODELS = _THREE_SITE_WATER_MODELS | _FOUR_SITE_WATER_MODELS | {"TIP5P"}
_WATER_MODEL_CHOICES = (
    "NONE", "OPC", "OPC3", "SPC", "SPCE", "TIP3P", "TIP4P",
    ("TIP4P-Ew", "TIP4PEW"), "TIP5P", "TIPS3P",
)
_BUNDLED_FORCE_FIELD_WATER_MODELS = {
    "AMBER03": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBER14SB": {"NONE", "OPC", "OPC3", "SPC", "SPCE", "TIP3P", "TIP4PEW"},
    "AMBER19SB": {"NONE", "OPC", "OPC3", "SPC", "SPCE", "TIP3P", "TIP4PEW"},
    "AMBER94": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBER96": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBER99": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBER99SB": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBER99SBILDN": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "AMBERGS": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "CHARMM27": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP5P", "TIPS3P"},
    "CHARMM36": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
    "GROMOS43A1": {"NONE", "SPC", "SPCE"},
    "GROMOS43A2": {"NONE", "SPC", "SPCE"},
    "GROMOS45A3": {"NONE", "SPC", "SPCE"},
    "GROMOS53A5": {"NONE", "SPC", "SPCE"},
    "GROMOS53A6": {"NONE", "SPC", "SPCE"},
    "GROMOS54A7": {"NONE", "SPC", "SPCE"},
    "OPLSAA": {"NONE", "SPC", "SPCE", "TIP3P", "TIP4P", "TIP4PEW", "TIP5P"},
}
_PREFERRED_WATER_MODELS = {
    "AMBER19SB": "OPC",
    "OPLSAA": "TIP4P",
}
_TOPOLOGY_INCLUDE_RE = re.compile(
    r'^\s*#include\s+["<]([^">]+)[">]', re.IGNORECASE | re.MULTILINE)


def _normalise_water_model(water_model: str) -> str:
    return str(water_model).upper().replace("-", "").replace("_", "")


def _normalise_force_field_id(force_field: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(force_field).upper())


def _supported_water_models(force_field: str) -> set[str] | None:
    """Return bundled pdb2gmx water choices, or None for a custom force field."""
    supported = _BUNDLED_FORCE_FIELD_WATER_MODELS.get(
        _normalise_force_field_id(force_field))
    return set(supported) if supported is not None else None


def _validate_force_field_water_model(force_field: str, water_model: str) -> None:
    """Reject unsupported built-in force-field/water combinations early."""
    supported = _supported_water_models(force_field)
    selected = _normalise_water_model(water_model)
    if supported is not None and selected not in supported:
        choices = ", ".join(
            value for value in ("NONE", "OPC", "OPC3", "SPC", "SPCE",
                                "TIP3P", "TIP4P", "TIP4PEW", "TIP5P", "TIPS3P")
            if value in supported)
        raise ValueError(
            f"Water model '{water_model}' is not supported by bundled force field "
            f"'{force_field}'. Choose one of: {choices}.")


def _water_choice_value(choice: str | tuple[str, str]) -> str:
    return choice[1] if isinstance(choice, tuple) else choice


def _solvent_configuration_for_water_model(water_model: str) -> str:
    """Map the topology water model to a coordinate template with equal sites."""
    normalized = _normalise_water_model(water_model)
    if normalized in _THREE_SITE_WATER_MODELS:
        return "spc216.gro"
    if normalized in _FOUR_SITE_WATER_MODELS:
        return "tip4p.gro"
    if normalized == "TIP5P":
        return "tip5p.gro"
    if normalized == "NONE":
        raise ValueError(
            "Solvation is unavailable when the topology water model is NONE. "
            "Generate the topology with a water model first."
        )
    raise ValueError(f"Unsupported water model for solvation: {water_model}.")


def _validate_topology_water_model(topology_path: str, water_model: str) -> None:
    """Ensure the already-generated topology uses the selected water model."""
    with open(topology_path, encoding="utf-8", errors="replace") as handle:
        topology = handle.read()
    included_models = {
        _normalise_water_model(os.path.splitext(os.path.basename(path))[0])
        for path in _TOPOLOGY_INCLUDE_RE.findall(topology)
    } & _KNOWN_WATER_MODELS
    selected_model = _normalise_water_model(water_model)
    if selected_model not in included_models:
        found = ", ".join(sorted(included_models)) or "no recognized water model"
        raise ValueError(
            f"Selected water model is {water_model}, but {os.path.basename(topology_path)} "
            f"contains {found}. Regenerate the topology or restore the matching "
            "water-model selection before solvation."
        )


def on_water_model_change(water_model: str) -> GradioUpdate:
    """Keep the solvent-coordinate field synchronized with the water topology."""
    try:
        return gr.update(value=_solvent_configuration_for_water_model(water_model))
    except ValueError:
        return gr.update(value=None)


def _minimum_box_padding_nm(force_field: str | None) -> float:
    normalized = str(force_field or "").lower()
    if normalized.startswith("gromos"):
        return 1.4
    if normalized.startswith("charmm"):
        return 1.2
    return 1.0


def on_force_field_change_for_box(force_field: str,
                                  current_distance: float) -> GradioUpdate:
    """Raise the box slider floor to the force family's nonbonded cutoff."""
    minimum = _minimum_box_padding_nm(force_field)
    try:
        distance = float(current_distance)
    except (TypeError, ValueError):
        raise ValueError("Box padding must be a finite number.") from None
    if not math.isfinite(distance):
        raise ValueError("Box padding must be a finite number.")
    return gr.update(minimum=minimum, value=max(distance, minimum))


def on_force_field_change(force_field: str, current_distance: float,
                          current_water_model: str) -> tuple[GradioUpdate, ...]:
    """Synchronize cutoff, supported waters, and solvent coordinates."""
    supported = _supported_water_models(force_field)
    choices = [
        choice for choice in _WATER_MODEL_CHOICES
        if supported is None
        or _normalise_water_model(_water_choice_value(choice)) in supported
    ]
    current = _normalise_water_model(current_water_model)
    allowed = ({_normalise_water_model(_water_choice_value(choice))
                for choice in choices})
    preferred = _PREFERRED_WATER_MODELS.get(
        _normalise_force_field_id(force_field))
    if current not in allowed:
        current = next(candidate for candidate in (
            preferred, "TIP3P", "SPC", "NONE")
            if candidate is not None and candidate in allowed)
    elif current == "TIP3P" and preferred in allowed:
        current = preferred
    return (
        on_force_field_change_for_box(force_field, current_distance),
        gr.update(choices=choices, value=current),
        on_water_model_change(current),
    )


def _validate_standard_time_step(time_step: float) -> float:
    """Reject timesteps that require hydrogen-mass repartitioning support."""
    value = float(time_step)
    if not (0 < value <= MAX_STANDARD_TIME_STEP_PS):
        raise ValueError(
            f"Time step must be greater than 0 and no more than "
            f"{MAX_STANDARD_TIME_STEP_PS:.3f} ps because this workflow does not "
            "apply hydrogen-mass repartitioning (HMR)."
        )
    return value


def _normalise_max_warnings(max_warnings: int) -> int:
    """Validate the deliberately small expert-only grompp override."""
    if isinstance(max_warnings, bool) or not isinstance(max_warnings, numbers.Real):
        raise ValueError("Max Warnings must be an integer from 0 to 10.")
    numeric = float(max_warnings)
    if (not math.isfinite(numeric) or not numeric.is_integer()
            or numeric < 0 or numeric > 10):
        raise ValueError("Max Warnings must be an integer from 0 to 10.")
    return int(numeric)


def _grompp_success(message: str, max_warnings: int,
                     continuation_warning: str | None = None) -> tuple[str, str]:
    """Return a success message and its severity colour for a grompp run."""
    warnings = []
    if max_warnings > 0:
        warnings.append(
            f"Expert override enabled: grompp was allowed to bypass up to "
            f"{max_warnings} warning(s); review its output before running MD."
        )
    if continuation_warning:
        warnings.append(continuation_warning)
    if warnings:
        return message + " " + " ".join(warnings), "orange"
    return message, "green"


def _validate_grompp_inputs(working_directory_path: str,
                             parameter_file_name: str,
                             topology_file_name: str,
                             force_field: str | None) -> tuple[str, str]:
    """Validate the UI choice and the selected MDP against the real topology."""
    parameter_path = os.path.join(working_directory_path, parameter_file_name)
    topology_path = os.path.join(working_directory_path, topology_file_name)
    if force_field is not None:
        validate_topology_force_field(topology_path, force_field)
    # This check is deliberately unconditional: a custom/edited MDP can be
    # scientifically incompatible even when no current UI force field is given.
    validate_mdp_topology_compatibility(parameter_path, topology_path)
    return parameter_path, topology_path


def _custom_force_field_warning(topology_path: str) -> str | None:
    """Explain the policy boundary when a job-local custom family is used."""
    try:
        force_field = get_topology_force_field_name(topology_path)
    except (OSError, ValueError):
        # Some callback unit tests replace the validator with a fixture tuple.
        # A real successful validation always leaves a readable topology here.
        return None
    if force_field and get_force_field_family(force_field) is None:
        return (
            f"Custom force field '{force_field}' detected: automatic "
            "family-specific cutoff compatibility checks are unavailable; "
            "you are responsible for verifying the MDP against that force field.")
    return None


def _publish_staged_files_unlocked(staged_files: Sequence[tuple[str, str]],
                                   remove_files: Sequence[str] = ()) -> None:
    """Transactionally replace a small set of related output files."""
    for staged_path, _ in staged_files:
        if (os.path.islink(staged_path)
                or not (os.path.isfile(staged_path)
                        or os.path.isdir(staged_path))):
            raise FileNotFoundError(
                f"Expected command output was not created: "
                f"{os.path.basename(staged_path)}"
            )

    final_paths = [final_path for _, final_path in staged_files]
    destinations = [*final_paths, *remove_files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Related output and removal paths must all be different.")
    # Never let a client-supplied output basename turn an existing directory,
    # symlink, socket, or FIFO into a disposable publish backup.
    for staged_path, destination in staged_files:
        if not os.path.lexists(destination):
            continue
        staged_is_directory = os.path.isdir(staged_path)
        destination_is_directory = os.path.isdir(destination)
        # ACPYPE's diagnostic directory is an intentional published artifact.
        # No other output field may replace a directory.
        safe_directory_replacement = (
            staged_is_directory and destination_is_directory
            and destination.lower().endswith(".acpype"))
        if (os.path.islink(destination)
                or (not os.path.isfile(destination)
                    and not safe_directory_replacement)
                or staged_is_directory != destination_is_directory):
            raise ValueError(
                "Refusing to replace non-regular job path: "
                f"{os.path.basename(destination)}")
    for destination in remove_files:
        if (os.path.lexists(destination)
                and (os.path.islink(destination)
                     or not os.path.isfile(destination))):
            raise ValueError(
                "Refusing to remove non-regular job path: "
                f"{os.path.basename(destination)}")

    backups: list[tuple[str, str]] = []
    published: list[str] = []
    try:
        for final_path in [*final_paths, *remove_files]:
            if os.path.exists(final_path):
                descriptor, backup_path = tempfile.mkstemp(
                    prefix=".publish_backup_", dir=os.path.dirname(final_path))
                os.close(descriptor)
                os.remove(backup_path)
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
        for staged_path, final_path in staged_files:
            os.replace(staged_path, final_path)
            published.append(final_path)
    except Exception:
        for final_path in reversed(published):
            try:
                if os.path.isdir(final_path) and not os.path.islink(final_path):
                    shutil.rmtree(final_path)
                else:
                    os.remove(final_path)
            except OSError:
                pass
        for backup_path, final_path in reversed(backups):
            if os.path.exists(backup_path):
                os.replace(backup_path, final_path)
        raise
    finally:
        for backup_path, _ in backups:
            try:
                if os.path.isdir(backup_path) and not os.path.islink(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            except OSError:
                pass


def _publish_staged_files(staged_files: Sequence[tuple[str, str]],
                          remove_files: Sequence[str] = ()) -> None:
    """Publish a staged bundle while excluding every other job writer."""
    destinations = [final_path for _, final_path in staged_files] + list(remove_files)
    if not destinations:
        return
    directories = {os.path.realpath(os.path.dirname(path)) for path in destinations}
    if len(directories) != 1:
        raise ValueError("Related outputs must share one working directory.")
    with reserve_working_directory_maintenance(directories.pop()):
        _publish_staged_files_unlocked(staged_files, remove_files)


def _validate_gaff_amber_compatibility(
        protein_force_field: str | None,
        protein_topology_path: str | None = None) -> None:
    """Hard-block mixing ACPYPE/GAFF parameters with a non-AMBER protein FF."""
    if protein_force_field and not str(protein_force_field).lower().startswith("amber"):
        raise ValueError(
            "ACPYPE generates GAFF ligand parameters, which are only supported "
            "here with an AMBER-family protein force field; selected protein "
            f"force field is {protein_force_field}."
        )

    if protein_topology_path and os.path.isfile(protein_topology_path):
        with open(protein_topology_path, encoding="utf-8", errors="replace") as handle:
            topology = handle.read()
        match = _FORCE_FIELD_INCLUDE_RE.search(topology)
        if match:
            force_field_id = match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
            if not force_field_id.lower().startswith("amber"):
                raise ValueError(
                    f"The selected protein topology includes {force_field_id}.ff, "
                    "but ACPYPE generates GAFF ligand parameters. Regenerate the "
                    "protein topology with an AMBER-family force field."
                )


def _ligand_topology_uses_gaff(ligand_topology_path: str) -> bool:
    """Recognize ACPYPE/GAFF provenance without blocking unrelated ligand ITPs."""
    with open(ligand_topology_path, encoding="utf-8", errors="replace") as handle:
        header = handle.read(16384)
    return re.search(
        r"\b(?:ACPYPE|GAFF2?|General\s+Amber\s+Force\s+Field)\b",
        header,
        flags=re.IGNORECASE,
    ) is not None

def get_working_directories() -> list[str]:
    """Names of the job directories that already exist under ./data, sorted by name."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted((entry.name for entry in DATA_ROOT.iterdir() if entry.is_dir()), key=str.lower)

def get_files_in_working_directory(working_directory_path: str | None) -> list[str]:
    """Visible files in a job directory, hiding backups and tool scratch files.

    Sorted by name: os.listdir() order is arbitrary, and every file dropdown in
    the UI is filtered straight out of this list."""
    if working_directory_path is None or not os.path.isdir(working_directory_path):
        return []
    files = [f for f in os.listdir(working_directory_path) if not (f.startswith('#') or f.startswith(MMPBSA_SCRATCH_PREFIX) or f.endswith("Zone.Identifier") or os.path.isdir(os.path.join(working_directory_path, f)))]
    return sorted(files, key=str.lower)

def get_default_cpu_count() -> int:
    """Physical core count, used as the upper bound of the MPI rank slider."""
    return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)


def _validate_positive_integer_resource(value: int, label: str,
                                        maximum: int) -> int:
    """Enforce a UI resource limit again at the server callback boundary."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{label} must be a positive integer.")
    count = int(value)
    if count < 1:
        raise ValueError(f"{label} must be a positive integer.")
    if count > maximum:
        raise ValueError(
            f"{label} must not exceed this server's limit of {maximum}.")
    return count


def _validate_mdrun_resources(mpi_rank: int,
                              omp_threads: int) -> tuple[int, int]:
    """Bound individual mdrun settings and their combined CPU-thread demand."""
    mpi_rank = _validate_positive_integer_resource(
        mpi_rank, "MPI ranks", get_default_cpu_count())
    omp_threads = _validate_positive_integer_resource(
        omp_threads, "OpenMP threads", 128)
    logical_cpus = max(
        1, psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    requested_threads = mpi_rank * omp_threads
    if requested_threads > logical_cpus:
        raise ValueError(
            f"MPI ranks ({mpi_rank}) × OpenMP threads ({omp_threads}) requests "
            f"{requested_threads} CPU threads, but this server exposes only "
            f"{logical_cpus}. Reduce MPI Ranks or OpenMP Threads.")
    return mpi_rank, omp_threads

def on_open_working_directory(working_directory: str | None) -> tuple[Any, ...]:
    """Create or open a job directory under ./data and enable the file actions."""
    if working_directory is None or working_directory.strip() == "":
        gr.Warning("Please specify a working directory.")
        return None, None, None, None, None, None

    try:
        validate_file_name(working_directory, "working directory")
        working_directory_path = str((DATA_ROOT / working_directory).resolve())
        if DATA_ROOT not in Path(working_directory_path).parents:
            raise ValueError(
                "Invalid working directory: path must stay inside ./data/")
        os.makedirs(working_directory_path, exist_ok=True)
        files = get_files_in_working_directory(working_directory_path)
    except (OSError, ValueError) as exc:
        gr.Warning(str(exc))
        return None, None, None, None, None, None

    return gr.update(choices=get_working_directories(), value=working_directory), working_directory_path, files, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def on_file_list_change(working_directory_path: str, protein_structure_file_name: str,
                        ligand_structure_file_name: str, protein_topology_output_file_name: str,
                        ligand_output_file_name: str, protein_topology_output_topology_file_name: str,
                        merge_structures_output_file_name: str, box_output_file_name: str,
                        merge_topologies_output_file_name: str, solvation_output_file_name: str,
                        solvation_output_topology_file_name: str, generate_ions_parameter_file_name: str,
                        generate_ions_run_input_file_name: str, generate_ions_output_file_name: str,
                        generate_ions_output_topology_file_name: str, energy_minimization_parameter_file_name:
                        str, energy_minimization_run_input_file_name: str,
                        nvt_equilibration_parameter_file_name: str, nvt_equilibration_run_input_file_name: str,
                        npt_equilibration_parameter_file_name: str, npt_equilibration_run_input_file_name: str,
                        prod_md_parameter_file_name: str, prod_md_run_input_file_name: str,
                        make_mol_whole_output_traj_file_name: str, center_protein_output_traj_file_name: str,
                        fit_backbone_output_traj_file_name: str) -> tuple[Any, ...]:
    """Refresh the file table and re-point every file dropdown.

    Each argument is the current value of the matching output-name textbox, so a
    dropdown keeps pointing at the file the previous step just produced.
    """
    files = get_files_in_working_directory(working_directory_path)
    # Update the file dataframe
    file_info = []
    for f in files:
        file_path = os.path.join(working_directory_path, f)
        lower_name = f.lower()
        if lower_name.endswith(('.pdb', '.gro')):
            file_type = "Structure File"
        elif lower_name.endswith(('.top', '.itp')):
            file_type = "Topology File"
        elif lower_name.endswith('.mdp'):
            file_type = "Parameter File"
        elif lower_name.endswith('.tpr'):
            file_type = "Run Input File"
        elif lower_name.endswith('.log'):
            file_type = "Log File"
        elif lower_name.endswith('.edr'):
            file_type = "Energy File"
        elif lower_name.endswith(('.trr', '.xtc')):
            file_type = "Trajectory File"
        elif lower_name.endswith('.cpt'):
            file_type = "Checkpoint File"
        elif lower_name.endswith(('.csv', '.xvg')):
            file_type = "Data File"
        elif lower_name.endswith('.ndx'):
            file_type = "Index File"
        else:
            file_type = "Other File"
        modified_timestamp = os.path.getmtime(file_path)
        file_info.append((modified_timestamp, [f, file_type, time.ctime(modified_timestamp)]))
    file_info.sort(key=lambda item: item[0], reverse=True)
    file_df = pd.DataFrame((row for _, row in file_info), columns=["File", "Type", "Modified"])

    # Filter structure and text files
    structure_files = [f for f in files if f.lower().endswith(('.pdb', '.gro'))]
    gro_files = [f for f in files if f.lower().endswith('.gro')]
    # A system topology and a molecule include are different GROMACS contracts.
    # Keeping them separate prevents an .itp from reaching grompp's -p input and
    # prevents a complete .top from being treated as the ligand molecule include.
    topology_files = [f for f in files if f.lower().endswith('.top')]
    ligand_topology_files = [f for f in files if f.lower().endswith('.itp')]
    parameter_files = [f for f in files if f.lower().endswith('.mdp')]
    run_input_files = [f for f in files if f.lower().endswith('.tpr')]
    checkpoint_files = [f for f in files if f.lower().endswith('.cpt')]
    # Both GROMACS and MDAnalysis accept compressed XTC and full-precision TRR.
    trajectory_files = [f for f in files if f.lower().endswith(('.xtc', '.trr'))]
    viewer_trajectory_files = trajectory_files
    # Companion result paths are fixed, so an arbitrary .dat summary cannot be
    # paired safely. Offer only gmx_MMPBSA's canonical summary. A legacy summary
    # under mmpbsa/ is exposed by the same basename; the loader migrates its full
    # artifact set into the job root before parsing it.
    results_files = []
    if MMPBSA_RESULTS_FILE_NAME in files:
        results_files.append(MMPBSA_RESULTS_FILE_NAME)
    else:
        try:
            legacy_summary = os.path.join(
                _legacy_mmpbsa_directory(working_directory_path),
                MMPBSA_RESULTS_FILE_NAME,
            )
            if os.path.isfile(legacy_summary):
                results_files.append(MMPBSA_RESULTS_FILE_NAME)
        except (OSError, ValueError):
            pass

    # Update protein topology input file name dropdown
    if protein_structure_file_name in structure_files:
        protein_topology_input_file_name_value = protein_structure_file_name
    else:
        protein_topology_input_file_name_value = structure_files[0] if structure_files else None

    # Update ligand topology input file name dropdown
    if ligand_structure_file_name in structure_files:
        ligand_topology_input_file_name_value = ligand_structure_file_name
    else:
        ligand_topology_input_file_name_value = structure_files[0] if structure_files else None
    
    # Update merge structure protein input file name dropdown
    if protein_topology_output_file_name in gro_files:
        merge_structure_protein_input_file_name_value = protein_topology_output_file_name
    else:
        merge_structure_protein_input_file_name_value = gro_files[0] if gro_files else None
    
    # Update merge structure ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.gro" in gro_files:
        merge_structure_ligand_input_file_name_value = f"{ligand_output_file_name}_GMX.gro"
    else:
        merge_structure_ligand_input_file_name_value = gro_files[0] if gro_files else None

    # Update merge topology protein input file name dropdown
    if protein_topology_output_topology_file_name in topology_files:
        merge_topology_protein_input_file_name_value = protein_topology_output_topology_file_name
    else:
        merge_topology_protein_input_file_name_value = topology_files[0] if topology_files else None
    
    # Update merge topology ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.itp" in ligand_topology_files:
        merge_topology_ligand_input_file_name_value = f"{ligand_output_file_name}_GMX.itp"
    else:
        merge_topology_ligand_input_file_name_value = (ligand_topology_files[0]
                                                       if ligand_topology_files else None)

    # Update box input file name dropdown
    if merge_structures_output_file_name in structure_files:
        box_input_file_name_value = merge_structures_output_file_name
    else:
        box_input_file_name_value = structure_files[0] if structure_files else None

    # Update solvation input file dropdown
    if box_output_file_name in structure_files:
        solvation_input_file_name_value = box_output_file_name
    else:
        solvation_input_file_name_value = structure_files[0] if structure_files else None

    # Update solvation input topology file dropdown
    if merge_topologies_output_file_name in topology_files:
        solvation_input_topology_file_name_value = merge_topologies_output_file_name
    else:
        solvation_input_topology_file_name_value = topology_files[0] if topology_files else None

    # Update generate ions input file dropdown
    if solvation_output_file_name in structure_files:
        generate_ions_input_file_name_value = solvation_output_file_name
    else:
        generate_ions_input_file_name_value = structure_files[0] if structure_files else None

    # Update generate ions input topology file dropdown
    if solvation_output_topology_file_name in topology_files:
        generate_ions_input_topology_file_name_value = solvation_output_topology_file_name
    else:
        generate_ions_input_topology_file_name_value = topology_files[0] if topology_files else None

    # Update generate ions parameter file dropdown
    if generate_ions_parameter_file_name in parameter_files:
        generate_ions_parameter_file_name_value = generate_ions_parameter_file_name
    else:
        generate_ions_parameter_file_name_value = parameter_files[0] if parameter_files else None

    # Update generate ions run input file dropdown
    if generate_ions_run_input_file_name in run_input_files:
        generate_ions_run_input_file_name_value = generate_ions_run_input_file_name
    else:
        generate_ions_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update energy minimization input file dropdown
    if generate_ions_output_file_name in structure_files:
        energy_minimization_input_file_name_value = generate_ions_output_file_name
    else:
        energy_minimization_input_file_name_value = structure_files[0] if structure_files else None

    # Update energy minimization run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        energy_minimization_input_topology_file_name_value = generate_ions_output_topology_file_name
    else:
        energy_minimization_input_topology_file_name_value = topology_files[0] if topology_files else None

    # Update energy minimization parameter file dropdown
    if energy_minimization_parameter_file_name in parameter_files:
        energy_minimization_parameter_file_name_value = energy_minimization_parameter_file_name
    else:
        energy_minimization_parameter_file_name_value = parameter_files[0] if parameter_files else None

    # Update energy minimization run input file dropdown
    if energy_minimization_run_input_file_name in run_input_files:
        energy_minimization_run_input_file_name_value = energy_minimization_run_input_file_name
    else:
        energy_minimization_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update nvt equilibration input file dropdown
    energy_minimization_output_structure = (os.path.splitext(energy_minimization_run_input_file_name)[0] + ".gro"
                                            if energy_minimization_run_input_file_name else "")
    if energy_minimization_run_input_file_name in run_input_files and energy_minimization_output_structure in structure_files:
        nvt_equilibration_input_file_name_value = energy_minimization_output_structure
    else:
        nvt_equilibration_input_file_name_value = structure_files[0] if structure_files else None

    # Update nvt equilibration run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        nvt_equilibration_input_topology_file_name_value = generate_ions_output_topology_file_name
    else:
        nvt_equilibration_input_topology_file_name_value = topology_files[0] if topology_files else None

    # Update nvt equilibration parameter file dropdown
    if nvt_equilibration_parameter_file_name in parameter_files:
        nvt_equilibration_parameter_file_name_value = nvt_equilibration_parameter_file_name
    else:
        nvt_equilibration_parameter_file_name_value = parameter_files[0] if parameter_files else None

    # Update nvt equilibration run input file dropdown
    if nvt_equilibration_run_input_file_name in run_input_files:
        nvt_equilibration_run_input_file_name_value = nvt_equilibration_run_input_file_name
    else:
        nvt_equilibration_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update npt equilibration input file dropdown
    nvt_output_structure = (os.path.splitext(nvt_equilibration_run_input_file_name)[0] + ".gro"
                            if nvt_equilibration_run_input_file_name else "")
    if nvt_equilibration_run_input_file_name in run_input_files and nvt_output_structure in structure_files:
        npt_equilibration_input_file_name_value = nvt_output_structure
    else:
        npt_equilibration_input_file_name_value = structure_files[0] if structure_files else None
    
    # Update npt equilibration run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        npt_equilibration_input_topology_file_name_value = generate_ions_output_topology_file_name
    else:
        npt_equilibration_input_topology_file_name_value = topology_files[0] if topology_files else None
    
    # Update npt equilibration parameter file dropdown
    if npt_equilibration_parameter_file_name in parameter_files:
        npt_equilibration_parameter_file_name_value = npt_equilibration_parameter_file_name
    else:
        npt_equilibration_parameter_file_name_value = parameter_files[0] if parameter_files else None

    # Update npt equilibration run input file dropdown
    if npt_equilibration_run_input_file_name in run_input_files:
        npt_equilibration_run_input_file_name_value = npt_equilibration_run_input_file_name
    else:
        npt_equilibration_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update production MD input file dropdown
    npt_output_structure = (os.path.splitext(npt_equilibration_run_input_file_name)[0] + ".gro"
                            if npt_equilibration_run_input_file_name else "")
    if npt_equilibration_run_input_file_name in run_input_files and npt_output_structure in structure_files:
        prod_md_input_file_name_value = npt_output_structure
    else:
        prod_md_input_file_name_value = structure_files[0] if structure_files else None
    
    # Update production MD run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        prod_md_input_topology_file_name_value = generate_ions_output_topology_file_name
    else:
        prod_md_input_topology_file_name_value = topology_files[0] if topology_files else None
    
    # Update production MD parameter file dropdown
    if prod_md_parameter_file_name in parameter_files:
        prod_md_parameter_file_name_value = prod_md_parameter_file_name
    else:
        prod_md_parameter_file_name_value = parameter_files[0] if parameter_files else None

    # Update production MD run input file dropdown
    if prod_md_run_input_file_name in run_input_files:
        prod_md_run_input_file_name_value = prod_md_run_input_file_name
    else:
        prod_md_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update production MD checkpoint file dropdown
    production_checkpoint = (
        os.path.splitext(prod_md_run_input_file_name_value)[0] + ".cpt"
        if prod_md_run_input_file_name_value else "")
    if production_checkpoint in checkpoint_files:
        prod_md_checkpoint_file_name_value = production_checkpoint
    else:
        # Never preselect another run's checkpoint. The user can select another
        # TPR/checkpoint pair explicitly, and the resume callback checks it.
        prod_md_checkpoint_file_name_value = None

    # Update fix trajectory run input file dropdown
    if prod_md_run_input_file_name in run_input_files:
        fix_traj_run_input_file_name_value = prod_md_run_input_file_name
    else:
        fix_traj_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update make molecule whole input trajectory file dropdown
    production_trajectory = (os.path.splitext(prod_md_run_input_file_name)[0] + ".xtc"
                             if prod_md_run_input_file_name else "")
    if prod_md_run_input_file_name in run_input_files and production_trajectory in trajectory_files:
        make_mol_whole_input_traj_file_name_value = production_trajectory
    else:
        make_mol_whole_input_traj_file_name_value = trajectory_files[0] if trajectory_files else None

    # Update center protein input trajectory file dropdown
    if make_mol_whole_output_traj_file_name in trajectory_files:
        center_protein_input_traj_file_name_value = make_mol_whole_output_traj_file_name
    else:
        center_protein_input_traj_file_name_value = trajectory_files[0] if trajectory_files else None

    # Update fit backbone input trajectory file dropdown
    if center_protein_output_traj_file_name in trajectory_files:
        fit_backbone_input_traj_file_name_value = center_protein_output_traj_file_name
    else:
        fit_backbone_input_traj_file_name_value = trajectory_files[0] if trajectory_files else None

    # Update analysis input file dropdown
    production_structure = (os.path.splitext(prod_md_run_input_file_name)[0] + ".gro"
                            if prod_md_run_input_file_name else "")
    if prod_md_run_input_file_name in run_input_files and production_structure in structure_files:
        analysis_structure_file_name_value = production_structure
    else:
        analysis_structure_file_name_value = structure_files[0] if structure_files else None

    # Update analysis input trajectory file dropdown
    if fit_backbone_output_traj_file_name in trajectory_files:
        analysis_input_traj_file_name_value = fit_backbone_output_traj_file_name
    else:
        production_trajectory_candidates = []
        for run_input_name in (prod_md_run_input_file_name,
                               prod_md_run_input_file_name_value):
            if run_input_name:
                stem = os.path.splitext(run_input_name)[0]
                production_trajectory_candidates.extend(
                    (stem + ".xtc", stem + ".trr"))
        analysis_input_traj_file_name_value = next(
            (name for name in production_trajectory_candidates
             if name in trajectory_files),
            trajectory_files[0] if trajectory_files else None)

    # The only safe summary name is the one whose companion artifact names are
    # defined above (and used by the loader).
    mmpbsa_results_file_name_value = (MMPBSA_RESULTS_FILE_NAME
                                      if results_files else None)

    return file_df, \
        gr.update(choices=structure_files, value=protein_topology_input_file_name_value), \
        gr.update(choices=structure_files, value=ligand_topology_input_file_name_value), \
        gr.update(choices=gro_files, value=merge_structure_protein_input_file_name_value), \
        gr.update(choices=gro_files, value=merge_structure_ligand_input_file_name_value), \
        gr.update(choices=topology_files, value=merge_topology_protein_input_file_name_value), \
        gr.update(choices=ligand_topology_files, value=merge_topology_ligand_input_file_name_value), \
        gr.update(choices=structure_files, value=box_input_file_name_value), \
        gr.update(choices=structure_files, value=solvation_input_file_name_value), \
        gr.update(choices=topology_files, value=solvation_input_topology_file_name_value), \
        gr.update(choices=structure_files, value=generate_ions_input_file_name_value), \
        gr.update(choices=topology_files, value=generate_ions_input_topology_file_name_value), \
        gr.update(choices=parameter_files, value=generate_ions_parameter_file_name_value), \
        gr.update(choices=run_input_files, value=generate_ions_run_input_file_name_value), \
        gr.update(choices=structure_files, value=energy_minimization_input_file_name_value), \
        gr.update(choices=topology_files, value=energy_minimization_input_topology_file_name_value), \
        gr.update(choices=parameter_files, value=energy_minimization_parameter_file_name_value), \
        gr.update(choices=run_input_files, value=energy_minimization_run_input_file_name_value), \
        gr.update(choices=structure_files, value=nvt_equilibration_input_file_name_value), \
        gr.update(choices=topology_files, value=nvt_equilibration_input_topology_file_name_value), \
        gr.update(choices=parameter_files, value=nvt_equilibration_parameter_file_name_value), \
        gr.update(choices=run_input_files, value=nvt_equilibration_run_input_file_name_value), \
        gr.update(choices=structure_files, value=npt_equilibration_input_file_name_value), \
        gr.update(choices=topology_files, value=npt_equilibration_input_topology_file_name_value), \
        gr.update(choices=parameter_files, value=npt_equilibration_parameter_file_name_value), \
        gr.update(choices=run_input_files, value=npt_equilibration_run_input_file_name_value), \
        gr.update(choices=structure_files, value=prod_md_input_file_name_value), \
        gr.update(choices=topology_files, value=prod_md_input_topology_file_name_value), \
        gr.update(choices=parameter_files, value=prod_md_parameter_file_name_value), \
        gr.update(choices=run_input_files, value=prod_md_run_input_file_name_value), \
        gr.update(choices=checkpoint_files, value=prod_md_checkpoint_file_name_value), \
        gr.update(choices=run_input_files, value=fix_traj_run_input_file_name_value), \
        gr.update(choices=trajectory_files, value=make_mol_whole_input_traj_file_name_value), \
        gr.update(choices=trajectory_files, value=center_protein_input_traj_file_name_value), \
        gr.update(choices=trajectory_files, value=fit_backbone_input_traj_file_name_value), \
        gr.update(choices=structure_files, value=analysis_structure_file_name_value), \
        gr.update(choices=trajectory_files, value=analysis_input_traj_file_name_value), \
        gr.update(choices=structure_files, value=analysis_structure_file_name_value), \
        gr.update(choices=viewer_trajectory_files, value=analysis_input_traj_file_name_value), \
        gr.update(choices=run_input_files, value=fix_traj_run_input_file_name_value), \
        gr.update(choices=topology_files, value=prod_md_input_topology_file_name_value), \
        gr.update(choices=results_files, value=mmpbsa_results_file_name_value)

def on_select_file(evt: gr.SelectData) -> tuple[Any, ...]:
    """Route the clicked file row to the structure or text viewer state."""
    selected_file_name = evt.row_value[0]
    lower_name = selected_file_name.lower()
    if lower_name.endswith(('.pdb', '.gro')):
        return selected_file_name, selected_file_name, None, gr.update(interactive=True)
    elif lower_name.endswith(('.top', '.itp', '.mdp', '.log', '.txt', '.dat', '.xvg', '.csv', '.ndx')):
        return selected_file_name, None, selected_file_name, gr.update(interactive=True)
    return selected_file_name, None, None, gr.update(interactive=True)

def on_selected_structure_file_state_change(state: str | None) -> tuple[GradioUpdate, GradioUpdate]:
    """Enable the View Structure button and reveal the accordion holding it."""
    # Open the accordion the button lives in, so a selected file is one click away.
    # Only ever open it: closing would collapse a viewer the user is reading just
    # because they clicked a row of a different type.
    return gr.update(interactive=(state is not None)), gr.update(open=True) if state is not None else gr.update()

def on_selected_text_file_state_change(state: str | None) -> tuple[GradioUpdate, GradioUpdate]:
    """Enable the View Text File button and reveal the accordion holding it."""
    return gr.update(interactive=(state is not None)), gr.update(open=True) if state is not None else gr.update()

def on_reset_working_directory_ui() -> tuple[Any, ...]:
    """Clear job-specific browser and analysis state before opening another job.

    Gradio states survive ordinary component refreshes.  Without an explicit reset,
    selecting ``topol.top`` in one job and then opening another left the old filename
    and editor contents live; Save or Delete could consequently act on a same-named
    file in the newly opened job.
    """
    return (
        # File selection and viewer actions.
        None,
        None,
        None,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(interactive=False),
        gr.update(label="Text File Viewer", value="", interactive=False),
        gr.update(interactive=False),
        gr.update(value=""),
        # Standard analysis data and plots.
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        None,
        gr.update(value=None),
        # MM-PBSA summary, companion plots and decomposition.
        None,
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        None,
        gr.update(value=None),
        # Never carry a Stop label from the job that was just left. Timers handle
        # any subsequent attachment to a process actually owned by the new job.
        gr.update(value="Start", variant="primary"),
        gr.update(value="Start", variant="primary"),
        gr.update(value="Start", variant="primary"),
        gr.update(value="Start", variant="primary"),
        gr.update(value="Start", variant="primary"),
    )

def on_open_working_directory_and_reset_ui(working_directory: str | None) -> tuple[Any, ...]:
    """Open a job and atomically discard UI values belonging to the prior job."""
    return (*on_open_working_directory(working_directory), *on_reset_working_directory_ui())

def on_delete_file(working_directory_path: str, selected_file_name: str | None) -> list[str]:
    """Delete the selected file and return the refreshed file list."""
    if selected_file_name is None:
        return get_files_in_working_directory(working_directory_path)
    
    file_path = os.path.join(working_directory_path, selected_file_name)
    try:
        with reserve_working_directory_maintenance(working_directory_path):
            os.remove(file_path)
        status = "File deleted successfully."
    except Exception as exc:
        status = "Error deleting file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_clean_working_directory(working_directory_path: str) -> list[str]:
    """Remove GROMACS backup files and Zone.Identifier leftovers."""
    try:
        with reserve_working_directory_maintenance(working_directory_path):
            files_to_clean = [f for f in os.listdir(working_directory_path) if f.startswith('#') or f.endswith("Zone.Identifier")]
            for f in files_to_clean:
                file_path = os.path.join(working_directory_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        status = "Working directory cleaned successfully."
    except Exception as exc:
        status = "Error cleaning working directory!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_view_protein_structure(working_directory_path: str, protein_file_name: str) -> tuple[str | None, str]:
    """Render a single frame with nglview and report the species it contains."""
    static_basename = None
    try:
        static_basename = static_asset_basename("complex_md_structure", working_directory_path)
        structure_path = STATIC_ROOT / f"{static_basename}.pdb"
        viewer_path = STATIC_ROOT / f"{static_basename}.html"
        # Representations follow whatever species the file actually contains, so the
        # ligand is picked up without hardcoding LIG and ions such as CU2P are drawn.
        protein_file_path, species = prepare_structure_viewer_file(
            os.path.join(working_directory_path, protein_file_name),
            str(structure_path),
        )

        # Create the NGL view widget
        view = nglview.show_structure_file(protein_file_path)
        add_species_representations_to_nglview(view, species)

        # Write the widget to HTML
        nglview.write_html(str(viewer_path), [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = (f'<iframe src="/static/{static_basename}.html?ts={timestamp}" '
                'height="800" width="600" title="NGL View"></iframe>')
        cleanup_stale_static_assets()

        return html, "<span style='color:green;'>" + get_species_legend(species) + "</span>"
    except Exception as exc:
        if static_basename is not None:
            remove_static_asset_bundle(static_basename)
        gr.Warning("Error!\n" + str(exc))
        return None, "<span style='color:red;'>Error loading structure!</span>"

def on_view_trajectory(working_directory_path: str, structure_file_name: str | None,
                       trajectory_file_name: str | None, selection: str,
                       max_frames: int) -> tuple[str | None, str | None]:
    """Reduce the trajectory, then return an iframe that animates it with NGL."""
    if structure_file_name is None or trajectory_file_name is None:
        gr.Warning("Please select both a structure file and a trajectory file.")
        return None, None

    static_basename = None
    try:
        static_basename = static_asset_basename("complex_md_trajectory", working_directory_path)
        info = write_trajectory_viewer_files(
            os.path.join(working_directory_path, structure_file_name),
            os.path.join(working_directory_path, trajectory_file_name),
            selection,
            max_frames,
            static_basename,
        )

        viewer_file_path = STATIC_ROOT / f"{static_basename}_view.html"
        timestamp = int(time.time())
        with open(viewer_file_path, 'w') as file:
            file.write(get_trajectory_viewer_html(static_basename, timestamp, info["frames"], info["species"]))
        cleanup_stale_static_assets()

        html = f'<iframe src="/static/{static_basename}_view.html?ts={timestamp}" height="800" width="600" title="NGL Trajectory View"></iframe>'
        status = (f"Showing {info['frames']} of {info['total_frames']} frames (every {info['stride']}), "
                  f"{info['n_atoms']} atoms. {get_species_legend(info['species'])}")

        return html, "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        if static_basename is not None:
            remove_static_asset_bundle(static_basename)
        gr.Warning("Error!\n" + str(exc))
        return None, "<span style='color:red;'>Error loading trajectory!</span>"

def on_view_text_file(working_directory_path: str,
                      text_file_name: str) -> tuple[GradioUpdate | None, GradioUpdate | None]:
    """Load a text file into the editor and enable saving it."""
    try:
        content = read_editable_text_file(
            working_directory_path, text_file_name)
        return gr.update(label=f"Text File Viewer - {text_file_name}", value=content, interactive=True), gr.update(interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, None

def on_save_text_file(working_directory_path: str, text_file_name: str | None,
                      text_content: str) -> list[str]:
    """Write the editor contents back to the file."""
    if text_file_name is None:
        gr.Warning("Please select a text file to save.")
        return get_files_in_working_directory(working_directory_path)
    
    try:
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_replace_editable_text_file(
                working_directory_path, text_file_name, text_content)
        status = "File saved successfully."
    except Exception as exc:
        status = "Error saving file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_upload_protein_structure_file(working_directory_path: str, protein_structure_file_name: str,
                                     protein_structure_file_path: str) -> tuple[list[str], str]:
    """Copy an uploaded protein structure into the job directory."""
    save_file_path = os.path.join(working_directory_path, protein_structure_file_name)
    temporary_path = None
    try:
        with reserve_working_directory_maintenance(working_directory_path):
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".upload_", suffix=os.path.splitext(protein_structure_file_name)[1],
                dir=working_directory_path)
            os.close(descriptor)
            shutil.copy2(protein_structure_file_path, temporary_path)
            os.replace(temporary_path, save_file_path)
            temporary_path = None

        status = "File uploaded successfully."
        return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        status = "Error uploading file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass

def on_upload_ligand_structure_file(working_directory_path: str, ligand_structure_file_name: str,
                                    ligand_residue_name: str,
                                    ligand_structure_file_path: str) -> tuple[list[str], str]:
    """Copy an uploaded ligand structure into the job directory as residue LIG.

    ``ligand_residue_name`` is what the ligand is called in the file being
    uploaded. It only needs changing when the file holds more than the ligand:
    if the name is absent from the file, every atom in it is treated as ligand,
    which is the usual case and covers files whose residue field is empty.
    """
    save_file_path = os.path.join(working_directory_path, ligand_structure_file_name)
    temporary_path = None
    try:
        with reserve_working_directory_maintenance(working_directory_path):
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".upload_", suffix=os.path.splitext(ligand_structure_file_name)[1],
                dir=working_directory_path)
            os.close(descriptor)
            shutil.copy2(ligand_structure_file_path, temporary_path)

            # Files in the wild name the molecule UNK, MOL, a component id, or leave
            # the field empty, but the analysis selects the ligand as "resname LIG".
            present = read_pdb_residue_names(temporary_path)
            replaced = rename_pdb_residues(temporary_path, LIGAND_RESNAME, ligand_residue_name)
            os.replace(temporary_path, save_file_path)
            temporary_path = None

        status = "File uploaded successfully."
        if replaced:
            status += (f" Residue name {', '.join(replaced)} renamed to "
                       f"{LIGAND_RESNAME} so the ligand stays selectable in the analysis.")
        elif present == [LIGAND_RESNAME]:
            status += f" The ligand is already residue {LIGAND_RESNAME}."
        else:
            status += (f" Nothing was renamed: this file contains "
                       f"{', '.join(present) or 'no atoms'}.")
        return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        status = "Error uploading file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    finally:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    
def on_generate_protein_topology(working_directory_path: str, input_file_name: str, output_file_name: str,
                                 output_topology_file_name: str, force_field: str, water_model: str,
                                 n_terminus: str, c_terminus: str) -> tuple[list[str], str]:
    """Run pdb2gmx, optionally choosing explicit N- and C-terminus patches."""
    try:
        _validate_force_field_water_model(force_field, water_model)
        # The bundled amberGS directory is case-sensitive; the other displayed
        # force-field names map to their lowercase directory identifiers.
        force_field_id = "amberGS" if force_field.lower() == "ambergs" else force_field.lower()
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".pdb2gmx_stage_", dir=working_directory_path) as stage_directory:
            stage_directory = os.path.abspath(stage_directory)
            shutil.copy2(
                os.path.join(working_directory_path, input_file_name),
                os.path.join(stage_directory, input_file_name),
            )
            for entry in os.scandir(working_directory_path):
                if entry.is_dir() and entry.name.lower().endswith(".ff"):
                    os.symlink(os.path.abspath(entry.path),
                               os.path.join(stage_directory, entry.name),
                               target_is_directory=True)

            cmd = [
                "gmx", "pdb2gmx",
                "-f", input_file_name,
                "-o", output_file_name,
                "-p", output_topology_file_name,
                "-i", "posre.itp",
                "-ff", force_field_id,
                "-water", water_model.lower(),
                "-ignh"
            ]

            select_termini = (n_terminus != DEFAULT_TERMINUS_CHOICE
                              or c_terminus != DEFAULT_TERMINUS_CHOICE)
            if select_termini:
                cmd.append("-ter")

            print(f"Running command (in {stage_directory}): {' '.join(cmd)}")

            if select_termini:
                answers, resolved_termini = resolve_terminus_selections(
                    cmd, stage_directory, n_terminus, c_terminus)

            if select_termini and answers is None:
                cmd.remove("-ter")
                run_checked_command(cmd, cwd=stage_directory)
                status = ("Topology generated successfully. This force field offers no "
                          "terminus selection, so its own default termini were applied.")
            elif select_termini:
                process = run_managed_command(
                    cmd, cwd=stage_directory, stdin_input=answers)
                stderr = process.stderr

                if process.returncode != 0:
                    raise Exception(stderr)

                status = ("Topology generated successfully. Termini: "
                          + ", ".join(resolved_termini) + ".")
            else:
                run_checked_command(cmd, cwd=stage_directory)
                status = "Topology generated successfully."

            staged_outputs = [
                os.path.join(stage_directory, output_file_name),
                os.path.join(stage_directory, output_topology_file_name),
            ]
            staged_outputs.extend(
                entry.path for entry in os.scandir(stage_directory)
                if entry.is_file() and entry.name.lower().endswith(".itp")
            )
            _publish_staged_files([
                (staged_path,
                 os.path.join(working_directory_path, os.path.basename(staged_path)))
                for staged_path in dict.fromkeys(staged_outputs)
            ])
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ligand_topology(working_directory_path: str, ligand_input_file_name: str,
                                ligand_output_file_name: str, ligand_charge: int,
                                ligand_charge_model: str,
                                ligand_force_field: str,
                                protein_force_field: str = "AMBER99SB-ILDN") -> tuple[list[str], str]:
    """Run acpype to parameterise the ligand with GAFF."""
    try:
        _validate_gaff_amber_compatibility(protein_force_field)
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".acpype_stage_", dir=working_directory_path) as stage_directory:
            cmd = [
                "acpype",
                "-i", os.path.join(working_directory_path, ligand_input_file_name),
                "-b", ligand_output_file_name,
                "-n", str(ligand_charge),
                "-c", ligand_charge_model,
                "-a", ligand_force_field,
            ]

            print(f"Running command: {' '.join(cmd)}")
            run_checked_command(cmd, cwd=stage_directory)

            # Publish the coordinate/topology pair, plus restraints when ACPYPE
            # generated them, only after every required artifact is present.
            ligand_dir = os.path.join(
                stage_directory, f'{ligand_output_file_name}.acpype')
            output_names = [
                f'{ligand_output_file_name}_GMX.gro',
                f'{ligand_output_file_name}_GMX.itp',
            ]
            for required_name in output_names:
                if not os.path.isfile(os.path.join(ligand_dir, required_name)):
                    raise FileNotFoundError(
                        f"Expected ACPYPE output was not created: {required_name}"
                    )
            ligand_posre_file_name = f'posre_{ligand_output_file_name}.itp'
            ligand_posre_source = os.path.join(ligand_dir, ligand_posre_file_name)
            ligand_posre_destination = os.path.join(
                working_directory_path, ligand_posre_file_name)
            if os.path.isfile(ligand_posre_source):
                output_names.append(ligand_posre_file_name)
            flat_stage_directory = os.path.join(stage_directory, ".flat_outputs")
            os.mkdir(flat_stage_directory)
            for file_name in output_names:
                shutil.copy2(os.path.join(ligand_dir, file_name),
                             os.path.join(flat_stage_directory, file_name))
            staged_artifacts = [
                (os.path.join(flat_stage_directory, file_name),
                 os.path.join(working_directory_path, file_name))
                for file_name in output_names
            ]
            # Preserve ACPYPE's detailed artifacts too; they are useful for
            # diagnostics and must correspond to the flat GRO/ITP pair.
            staged_artifacts.append((
                ligand_dir,
                os.path.join(working_directory_path,
                             f'{ligand_output_file_name}.acpype'),
            ))
            _publish_staged_files(
                staged_artifacts,
                remove_files=([] if os.path.isfile(ligand_posre_source)
                              else [ligand_posre_destination]),
            )

        status = "Topology generated successfully."
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_structures(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                        output_file: str,
                        ligand_topology_file: str | None = None) -> tuple[list[str], str]:
    """Combine protein and ligand coordinates into one complex structure."""
    try:
        if not (str(protein_input_file).lower().endswith(".gro")
                and str(ligand_input_file).lower().endswith(".gro")):
            raise ValueError("Both structure merge inputs must be GROMACS .gro files.")
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        ligand_topology_file_path = (
            os.path.join(working_directory_path, ligand_topology_file)
            if ligand_topology_file else None)
        with reserve_working_directory_maintenance(working_directory_path):
            warnings = merge_protein_ligand_structures(
                protein_input_file_path, ligand_input_file_path, output_file_path,
                ligand_topology_file_path)

        status = "Structure files merged successfully."
        color = "green"
        if isinstance(warnings, (list, tuple)) and warnings:
            status += " Warning: " + " ".join(str(warning) for warning in warnings)
            color = "orange"
    except Exception as exc:
        status = "Error merging structure files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def on_merge_topologies(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                        output_file: str,
                        protein_force_field: str = "AMBER99SB-ILDN",
                        ligand_structure_file: str | None = None) -> tuple[list[str], str]:
    """Combine the protein and ligand topologies into one complex topology."""
    try:
        if not str(protein_input_file).lower().endswith(".top"):
            raise ValueError("Protein topology input must be a complete .top file.")
        if not str(ligand_input_file).lower().endswith(".itp"):
            raise ValueError("Ligand topology input must be a molecule .itp file.")
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        ligand_structure_file_path = (
            os.path.join(working_directory_path, ligand_structure_file)
            if ligand_structure_file else None)
        with reserve_working_directory_maintenance(working_directory_path):
            if _ligand_topology_uses_gaff(ligand_input_file_path):
                _validate_gaff_amber_compatibility(
                    protein_force_field, protein_input_file_path)
            merge_protein_ligand_topologies(
                protein_input_file_path, ligand_input_file_path, output_file_path,
                ligand_structure_file_path)

        status = "Topology files merged successfully."
    except Exception as exc:
        status = "Error merging topology files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_topology(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                      output_file: str,
                      protein_force_field: str = "AMBER99SB-ILDN",
                      ligand_structure_file: str | None = None) -> tuple[list[str], str]:
    """Combine the protein and ligand topologies into one complex topology."""
    try:
        if not str(protein_input_file).lower().endswith(".top"):
            raise ValueError("Protein topology input must be a complete .top file.")
        if not str(ligand_input_file).lower().endswith(".itp"):
            raise ValueError("Ligand topology input must be a molecule .itp file.")
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        ligand_structure_file_path = (
            os.path.join(working_directory_path, ligand_structure_file)
            if ligand_structure_file else None)
        with reserve_working_directory_maintenance(working_directory_path):
            if _ligand_topology_uses_gaff(ligand_input_file_path):
                _validate_gaff_amber_compatibility(
                    protein_force_field, protein_input_file_path)
            merge_protein_ligand_topologies(
                protein_input_file_path, ligand_input_file_path, output_file_path,
                ligand_structure_file_path)

        status = "Topology files merged successfully."
    except Exception as exc:
        status = "Error merging topology files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_simulation_box(working_directory_path: str, input_file_name: str, output_file_name: str,
                               box_type: str, distance: float,
                               force_field: str | None = None) -> tuple[list[str], str]:
    """Run editconf to centre the solute in a box of the requested shape."""
    try:
        minimum_distance = _minimum_box_padding_nm(force_field)
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            raise ValueError("Box padding must be a finite number.") from None
        if not math.isfinite(distance):
            raise ValueError("Box padding must be a finite number.")
        if distance < minimum_distance:
            raise ValueError(
                f"Box padding must be at least {minimum_distance:.1f} nm for "
                f"the {force_field or 'selected'} force-field family."
            )
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".box_stage_", dir=working_directory_path) as stage_directory:
            staged_output = os.path.join(stage_directory, "boxed.gro")
            cmd = [
                "gmx", "editconf",
                "-f", os.path.join(working_directory_path, input_file_name),
                "-o", staged_output,
                "-c",
                "-d", str(distance),
                "-bt", box_type
            ]

            print(f"Running command: {' '.join(cmd)}")

            run_checked_command(cmd, cwd=working_directory_path)
            _publish_staged_files([
                (staged_output,
                 os.path.join(working_directory_path, output_file_name)),
            ])
        status = "Simulation box generated successfully."
    except Exception as exc:
        status = "Error generating simulation box!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_solvate_protein(working_directory_path: str, input_file_name: str, output_file_name: str,
                       input_topology_file_name: str, output_topology_file_name: str,
                       solvent_configuration: str,
                       water_model: str | None = None) -> tuple[list[str], str]:
    """Run solvate to fill the box with the chosen solvent configuration."""
    try:
        if water_model is not None:
            expected_configuration = _solvent_configuration_for_water_model(water_model)
            if solvent_configuration != expected_configuration:
                raise ValueError(
                    f"Water model {water_model} requires {expected_configuration}; "
                    f"refusing mismatched solvent coordinates {solvent_configuration}."
                )
            _validate_topology_water_model(
                os.path.join(working_directory_path, input_topology_file_name),
                water_model,
            )

        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".solvate_stage_", dir=working_directory_path) as stage_directory:
            staged_structure = os.path.join(stage_directory, "solvated.gro")
            staged_topology = os.path.join(stage_directory, "solvated.top")
            shutil.copy2(
                os.path.join(working_directory_path, input_topology_file_name),
                staged_topology,
            )

            cmd = [
                "gmx", "solvate",
                "-cp", os.path.join(working_directory_path, input_file_name),
                "-cs", solvent_configuration,
                "-o", staged_structure,
                "-p", staged_topology,
            ]

            print(f"Running command: {' '.join(cmd)}")
            run_checked_command(cmd, cwd=working_directory_path)
            _publish_staged_files([
                (staged_structure, os.path.join(working_directory_path, output_file_name)),
                (staged_topology,
                 os.path.join(working_directory_path, output_topology_file_name)),
            ])
        status = "Protein solvated successfully."
    except Exception as exc:
        status = "Error solvating protein!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ions_mdp_file(working_directory_path: str, parameter_file_name: str,
                              force_field: str) -> tuple[list[str], str]:
    """Write the MDP used for the minimisation that precedes ion placement."""
    file_content = get_default_ion_addition_mdp_file_content(force_field=force_field)
    try:
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(file_path, file_content)
        status = "Ion addition parameter file generated successfully."
    except Exception as exc:
        status = "Error generating ion addition parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ions_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                              parameter_file_name: str, run_input_file_name: str,
                              max_warnings: int,
                              force_field: str | None = None) -> tuple[list[str], str]:
    """Run grompp to build the run input file that genion needs."""
    try:
        max_warnings = _normalise_max_warnings(max_warnings)
        parameter_path, topology_path = _validate_grompp_inputs(
            working_directory_path, parameter_file_name,
            input_topology_file_name, force_field)
        compatibility_warning = _custom_force_field_warning(topology_path)
        cmd = [
            "gmx", "grompp",
            "-f", parameter_path,
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", topology_path,
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        gromos_warning = run_grompp_with_gromos_warning_policy(
            cmd, working_directory_path, topology_path, max_warnings,
            runner=run_checked_command)
        compatibility_warning = " ".join(
            warning for warning in (compatibility_warning, gromos_warning)
            if warning) or None
        status, color = _grompp_success(
            "Ion addition run input file generated successfully.", max_warnings,
            compatibility_warning)
    except Exception as exc:
        status = "Error generating ion addition run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def on_add_ions_method_change(add_ions_method: str) -> tuple[GradioUpdate, ...]:
    """Show mode inputs while keeping ion valences available in both modes."""
    if add_ions_method == "Concentration":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:  # add_ions_method == "Number"
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def _find_sol_group(genion_cmd: Sequence[str], working_directory_path: str) -> str:
    """Detect the SOL group number genion offers, which depends on the topology.

    Group numbering is not fixed across force fields, so a probe run is parsed
    instead of assuming a well-known index."""
    with tempfile.TemporaryDirectory(
            prefix=".probe_genion_", dir=working_directory_path) as probe_directory:
        tmp_gro = os.path.join(probe_directory, "probe.gro")
        tmp_top = os.path.join(probe_directory, "probe.top")
        probe_cmd = list(genion_cmd)
        probe_cmd[probe_cmd.index("-o") + 1] = tmp_gro
        top_idx = probe_cmd.index("-p") + 1
        shutil.copy2(probe_cmd[top_idx], tmp_top)
        probe_cmd[top_idx] = tmp_top

        probe = run_managed_command(
            probe_cmd, cwd=working_directory_path, stdin_input="0\n")
        stderr_probe = probe.stderr

    sol_group = find_gmx_group_number(stderr_probe, "SOL")
    if sol_group is None:
        raise Exception(f"Could not find SOL group in genion output:\n{stderr_probe}")

    return sol_group

def on_add_ions(working_directory_path: str, run_input_file_name: str, output_file_name: str,
                input_topology_file_name: str, output_topology_file_name: str, cation_name: str,
                anion_name: str, add_ion_method: str, concentration: float, cation_charge: int,
                anion_charge: int, number_of_cations: int, number_of_anions: int,
                neutralize: bool) -> tuple[list[str], str]:
    """Run genion to neutralise the system and reach the requested ion content."""
    try:
        (add_ion_method, concentration, cation_charge, anion_charge,
         number_of_cations, number_of_anions,
         neutralize) = validate_ion_addition_parameters(
            add_ion_method, concentration, cation_charge, anion_charge,
            number_of_cations, number_of_anions, neutralize)
        cation_name, anion_name = validate_ion_species_charges(
            cation_name, cation_charge, anion_name, anion_charge)
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".genion_stage_", dir=working_directory_path) as stage_directory:
            staged_structure = os.path.join(stage_directory, "ions.gro")
            staged_topology = os.path.join(stage_directory, "ions.top")
            shutil.copy2(
                os.path.join(working_directory_path, input_topology_file_name),
                staged_topology,
            )

            cmd = [
                "gmx", "genion",
                "-s", os.path.join(working_directory_path, run_input_file_name),
                "-o", staged_structure,
                "-p", staged_topology,
                "-pname", cation_name,
                "-nname", anion_name,
                "-pq", str(cation_charge),
                "-nq", str(anion_charge),
            ]

            if neutralize:
                cmd.append("-neutral")

            if add_ion_method == "Concentration":
                cmd.extend(["-conc", str(concentration / 1000.0)])  # convert mM to M
            else:  # add_ion_method == "Number"
                cmd.extend(["-np", str(number_of_cations),
                            "-nn", str(number_of_anions)])

            print(f"Running command: {' '.join(cmd)}")

            sol_group = _find_sol_group(cmd, working_directory_path)
            process = run_managed_command(
                cmd, cwd=working_directory_path,
                stdin_input=f"{sol_group}\n")
            stderr = process.stderr

            if process.returncode != 0:
                raise Exception(stderr)
            validate_ionized_system_with_grompp(
                staged_structure, staged_topology, working_directory_path,
                runner=run_checked_command)
            _publish_staged_files([
                (staged_structure, os.path.join(working_directory_path, output_file_name)),
                (staged_topology,
                 os.path.join(working_directory_path, output_topology_file_name)),
            ])

        status = "Ions added successfully."
    except Exception as exc:
        status = "Error adding ions!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_energy_minimization_mdp_file(working_directory_path: str, parameter_file_name: str,
                                             force_field: str) -> tuple[list[str], str]:
    """Write the steepest-descent energy minimisation MDP."""
    file_content = get_default_energy_minimization_mdp_file_content(force_field=force_field)
    try:
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(file_path, file_content)
        status = "Energy minimization parameter file generated successfully."
    except Exception as exc:
        status = "Error generating energy minimization parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_energy_minimization_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                             parameter_file_name: str, run_input_file_name: str,
                                             max_warnings: int,
                                             force_field: str | None = None) -> tuple[list[str], str]:
    """Run grompp to build the energy minimisation run input file."""
    try:
        max_warnings = _normalise_max_warnings(max_warnings)
        parameter_path, topology_path = _validate_grompp_inputs(
            working_directory_path, parameter_file_name,
            input_topology_file_name, force_field)
        compatibility_warning = _custom_force_field_warning(topology_path)
        cmd = [
            "gmx", "grompp",
            "-f", parameter_path,
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", topology_path,
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        gromos_warning = run_grompp_with_gromos_warning_policy(
            cmd, working_directory_path, topology_path, max_warnings,
            runner=run_checked_command)
        compatibility_warning = " ".join(
            warning for warning in (compatibility_warning, gromos_warning)
            if warning) or None
        status, color = _grompp_success(
            "Energy minimization run input file generated successfully.", max_warnings,
            compatibility_warning)
    except Exception as exc:
        status = "Error generating energy minimization run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def on_run_energy_minimization(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                               omp_threads: int, use_gpu: bool) -> tuple[list[str], str]:
    """Run mdrun for energy minimisation and wait for it to finish.

    use_gpu is deliberately ignored: GROMACS cannot run PME on the GPU during
    energy minimisation, so this step always stays on the CPU."""
    job_key = None
    claimed = False
    try:
        mpi_rank, omp_threads = _validate_mdrun_resources(mpi_rank, omp_threads)
        base_name = os.path.splitext(run_input_file_name)[0]
        job_key = get_process_job_key(working_directory_path, base_name)
        claimed, _ = reserve_process_job(job_key)
        if not claimed:
            raise WorkingDirectoryBusyError(
                "Another process or file operation is already using this output."
            )
        reserve_process_resources(job_key, mpi_rank, omp_threads, False)

        # Every mdrun runs from the job directory with plain file names. -deffnm
        # would place its own outputs correctly either way, but the PDBs mdrun
        # dumps when constraints fail (step<n>b.pdb / step<n>c.pdb) have hardcoded
        # names and no flag to redirect them, so they follow the working directory.
        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ] + get_cpu_only_mdrun_options()

        run_checked_command(cmd, cwd=working_directory_path)
        status = "Energy minimization completed successfully."
    except Exception as exc:
        status = "Error during energy minimization!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    finally:
        if claimed:
            release_process_job(job_key)
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                           temperature: float, parameter_file_name: str,
                                           force_field: str) -> tuple[list[str], str]:
    """Write the restrained NVT equilibration MDP."""
    try:
        time_step = _validate_standard_time_step(time_step)
        file_content = get_default_nvt_equilibration_mdp_file_content(
            time_scale_ps=time_scale, time_step_ps=time_step,
            temperature=temperature, with_ligand=True, force_field=force_field)
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(file_path, file_content)
        status = "NVT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NVT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                           parameter_file_name: str, run_input_file_name: str,
                                           max_warnings: int,
                                           force_field: str | None = None) -> tuple[list[str], str]:
    """Run grompp to build the NVT run input file, with restraint references."""
    try:
        max_warnings = _normalise_max_warnings(max_warnings)
        parameter_path, topology_path = _validate_grompp_inputs(
            working_directory_path, parameter_file_name,
            input_topology_file_name, force_field)
        compatibility_warning = _custom_force_field_warning(topology_path)
        cmd = [
            "gmx", "grompp",
            "-f", parameter_path,
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", topology_path,
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        gromos_warning = run_grompp_with_gromos_warning_policy(
            cmd, working_directory_path, topology_path, max_warnings,
            runner=run_checked_command)
        compatibility_warning = " ".join(
            warning for warning in (compatibility_warning, gromos_warning)
            if warning) or None
        status, color = _grompp_success(
            "NVT equilibration run input file generated successfully.", max_warnings,
            compatibility_warning)
    except Exception as exc:
        status = "Error generating NVT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def sync_button_state(process_state: ProcessStateDict) -> GradioUpdate:
    """Keep a Run/Stop button label in step with the process state."""
    with process_state["lock"]:
        if process_state["running"]:
            return gr.update(value="Stop", variant="stop")
        else:
            return gr.update(value="Start", variant="primary")

def sync_process_state(working_directory_path: str,
                       process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Publish a long-running job's one-shot completion and new output files."""
    current_directory = (os.path.realpath(working_directory_path)
                         if working_directory_path else None)
    with process_state["lock"]:
        proc = process_state.get("proc") if process_state.get("running") else None
        associated_directory = process_state.get("working_directory")
    if (proc is not None and associated_directory
            and current_directory != associated_directory):
        clear_process_state_if_current(process_state, proc)
        return (gr.update(), gr.update(),
                gr.update(value="Start", variant="primary"))

    refresh_process_state(process_state)
    running, message, color, job_directory = consume_process_completion(process_state)
    button = (gr.update(value="Stop", variant="stop") if running
              else gr.update(value="Start", variant="primary"))
    if message is None:
        return gr.update(), gr.update(), button

    files = (get_files_in_working_directory(job_directory)
             if current_directory == job_directory else gr.update())
    return files, f"<span style='color:{color};'>{html.escape(message)}</span>", button


def _process_timer_update(process_state: ProcessStateDict) -> GradioUpdate:
    """Run the poller only while a process or completion notice needs attention."""
    with process_state["lock"]:
        active = bool(process_state.get("running")
                      or process_state.get("completion_pending"))
    return gr.update(active=active)


def _sync_process_state_with_timer(
        working_directory_path: str,
        process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Poll one process control and stop its timer once it is fully idle."""
    result = sync_process_state(working_directory_path, process_state)
    return (*result, _process_timer_update(process_state))


def _sync_shared_process_state_with_timer(
        working_directory_path: str,
        process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Poll the shared production job and keep both action buttons in sync."""
    files, status, button = sync_process_state(
        working_directory_path, process_state)
    return (files, status, button, dict(button),
            _process_timer_update(process_state))

def _claim_process_output(working_directory_path: str, output_prefix: str,
                          process_state: ProcessStateDict, job_name: str,
                          failure_hint: str | None = None) -> tuple[
                              str, bool, subprocess.Popen[str] | None]:
    """Reserve an output path, adopting a live run after a page refresh."""
    job_key = get_process_job_key(working_directory_path, output_prefix)
    claimed, active_proc = reserve_process_job(job_key)
    if not claimed and active_proc is not None:
        metadata = get_registered_process_metadata(job_key, active_proc)
        if metadata is not None:
            job_name, working_directory_path, failure_hint = metadata
        set_process_running(process_state, active_proc, job_key, job_name,
                            working_directory_path, failure_hint)
    return job_key, claimed, active_proc

def _activate_process(proc: subprocess.Popen[str], process_state: ProcessStateDict,
                      job_key: str, job_name: str, working_directory_path: str,
                      failure_hint: str | None = None) -> None:
    """Register a child, associate it with the UI, and start its watcher."""
    if os.name == "posix":
        # Every caller launches with start_new_session=True.
        try:
            setattr(proc, "_gromacs_webui_process_group", proc.pid)
        except (AttributeError, TypeError):
            pass
    register_process_job(job_key, proc, job_name, working_directory_path, failure_hint)
    set_process_running(process_state, proc, job_key, job_name,
                        working_directory_path, failure_hint)
    threading.Thread(target=watch_process, args=(proc, process_state, job_key),
                     daemon=True).start()

def _clean_up_failed_process_start(job_key: str | None,
                                   proc: subprocess.Popen[str] | None,
                                   process_state: ProcessStateDict) -> None:
    """Leave neither a running child nor a stale reservation after launch fails."""
    if proc is not None:
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)
    # Registration failures leave the reserved sentinel in place; clear it and
    # its resource admission without touching any replacement process.
    release_process_job(job_key)
    clear_process_state_if_current(process_state, proc)
    
def on_run_nvt_equilibration(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                             omp_threads: int, use_gpu: bool,
                             process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start NVT equilibration, or stop the run that is already in progress."""
    # ---------- STOP ----------
    proc, job_key = clear_process_state_for_directory(process_state, working_directory_path)
    if proc is not None:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)

        status = "NVT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    proc = None
    job_key = None
    try:
        mpi_rank, omp_threads = _validate_mdrun_resources(
            mpi_rank, omp_threads)
        base_name = os.path.splitext(run_input_file_name)[0]
        job_key, claimed, active_proc = _claim_process_output(
            working_directory_path, base_name, process_state, "NVT equilibration",
            f"See {base_name}.log for details.")
        if not claimed:
            if active_proc is not None:
                status = (f"{get_process_job_name(process_state)} is already running "
                          "for this output. This session is now attached to it; "
                          "click Stop to end it.")
                button = gr.update(value="Stop", variant="stop")
            else:
                status = "NVT equilibration is already starting for this output."
                button = gr.update(value="Start", variant="primary")
            return (get_files_in_working_directory(working_directory_path),
                    f"<span style='color:orange;'>{status}</span>", process_state, button)

        resource_status = reserve_process_resources(
            job_key, mpi_rank, omp_threads, use_gpu)

        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ] + get_mdrun_hardware_options(use_gpu, mpi_rank)

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True,
                                start_new_session=True)
        _activate_process(proc, process_state, job_key, "NVT equilibration",
                          working_directory_path, f"See {base_name}.log for details.")

        status = f"NVT equilibration started. {resource_status}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        _clean_up_failed_process_start(job_key, proc, process_state)

        status = f"Error during NVT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_generate_npt_equilibration_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                           temperature: float, pressure: float, parameter_file_name: str,
                                           force_field: str) -> tuple[list[str], str]:
    """Write the restrained NPT equilibration MDP."""
    try:
        time_step = _validate_standard_time_step(time_step)
        file_content = get_default_npt_equilibration_mdp_file_content(
            time_scale_ps=time_scale, time_step_ps=time_step,
            temperature=temperature, pressure=pressure, with_ligand=True,
            force_field=force_field)
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(file_path, file_content)
        status = "NPT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NPT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_npt_equilibration_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                           parameter_file_name: str, run_input_file_name: str,
                                           max_warnings: int,
                                           force_field: str | None = None) -> tuple[list[str], str]:
    """Run grompp to build the NPT run input file, with restraint references."""
    try:
        max_warnings = _normalise_max_warnings(max_warnings)
        parameter_path, topology_path = _validate_grompp_inputs(
            working_directory_path, parameter_file_name,
            input_topology_file_name, force_field)
        compatibility_warning = _custom_force_field_warning(topology_path)
        cmd = [
            "gmx", "grompp",
            "-f", parameter_path,
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", topology_path,
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]
        checkpoint_path = get_matching_checkpoint_path(working_directory_path, input_file_name)
        if checkpoint_path is not None:
            cmd.extend(["-t", checkpoint_path])
        
        print(f"Running command: {' '.join(cmd)}")

        gromos_warning = run_grompp_with_gromos_warning_policy(
            cmd, working_directory_path, topology_path, max_warnings,
            runner=run_checked_command)
        continuation_warning = " ".join(
            warning for warning in (compatibility_warning, gromos_warning)
            if warning) or None
        if checkpoint_path is None:
            checkpoint_warning = ("No matching checkpoint was found; GROMACS can read "
                                  "velocities from the input structure, but thermostat "
                                  "state cannot be carried over.")
            continuation_warning = " ".join(
                warning for warning in (continuation_warning, checkpoint_warning)
                if warning)
        status, color = _grompp_success(
            "NPT equilibration run input file generated successfully.",
            max_warnings,
            continuation_warning,
        )
    except Exception as exc:
        status = "Error generating NPT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def on_run_npt_equilibration(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                             omp_threads: int, use_gpu: bool,
                             process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start NPT equilibration, or stop the run that is already in progress."""
    # ---------- STOP ----------
    proc, job_key = clear_process_state_for_directory(process_state, working_directory_path)
    if proc is not None:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)

        status = "NPT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    proc = None
    job_key = None
    try:
        mpi_rank, omp_threads = _validate_mdrun_resources(
            mpi_rank, omp_threads)
        base_name = os.path.splitext(run_input_file_name)[0]
        job_key, claimed, active_proc = _claim_process_output(
            working_directory_path, base_name, process_state, "NPT equilibration",
            f"See {base_name}.log for details.")
        if not claimed:
            if active_proc is not None:
                status = (f"{get_process_job_name(process_state)} is already running "
                          "for this output. This session is now attached to it; "
                          "click Stop to end it.")
                button = gr.update(value="Stop", variant="stop")
            else:
                status = "NPT equilibration is already starting for this output."
                button = gr.update(value="Start", variant="primary")
            return (get_files_in_working_directory(working_directory_path),
                    f"<span style='color:orange;'>{status}</span>", process_state, button)

        resource_status = reserve_process_resources(
            job_key, mpi_rank, omp_threads, use_gpu)

        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ] + get_mdrun_hardware_options(use_gpu, mpi_rank)

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True,
                                start_new_session=True)
        _activate_process(proc, process_state, job_key, "NPT equilibration",
                          working_directory_path, f"See {base_name}.log for details.")

        status = f"NPT equilibration started. {resource_status}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        _clean_up_failed_process_start(job_key, proc, process_state)

        status = f"Error during NPT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_toggle_nnpot(nnpot_active: bool, nnpot_model_name: str = "ani2x") -> str:
    """Acknowledge the neural-network potential choice in the status line."""
    if not nnpot_active:
        return ""
    unavailable_reason = get_nnpot_unavailable_reason(nnpot_model_name)
    if unavailable_reason is not None:
        safe_reason = html.escape(unavailable_reason).replace("\n", "<br>")
        return f"<span style='color:red;'>{safe_reason}</span>"
    return ("<span style='color:green;'>Machine learning potential enabled. "
            "The selected model will be built when you generate the production MD parameter file.</span>")

def on_change_mdp_type(prod_md_mdp_type_radio: str) -> tuple[GradioUpdate, str]:
    """Switch the production MDP between an initial run and a continuation."""
    if prod_md_mdp_type_radio=="Initial":
        return gr.update(visible=False), "md_initial.mdp"
    else:
        return gr.update(visible=False), "md_continue.mdp"

def on_generate_prod_md_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                 temperature: float, pressure: float, mdp_type: str, random_seed: int,
                                 parameter_file_name: str, nnpot_active: bool, nnpot_model_name: str,
                                 nnpot_input_group: str, force_field: str) -> tuple[list[str], str]:
    """Write the production MD MDP, building the neural potential if requested."""
    try:
        time_step = _validate_standard_time_step(time_step)
    except Exception as exc:
        status = "Error generating production MD parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"

    if parameter_file_name is None or str(parameter_file_name).strip() == "":
        parameter_file_name = "md_initial.mdp" if mdp_type == "Initial" else "md_continue.mdp"

    # Build (or reuse) the requested NNPot model via the universal wrapper and
    # collect the matching nnpot-model-input* keywords before writing the MDP.
    nnpot_modelfile_path = None
    if nnpot_active:
        try:
            nnpot_modelfile_path = download_nnpot_model(nnpot_model_name)
        except Exception as exc:
            status = "Error generating NNPot model!\n" + str(exc)
            return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"

    try:
        file_content = get_default_prod_md_mdp_file_content(time_scale_ps=time_scale*1000, time_step_ps=time_step, temperature=temperature, pressure=pressure, mdp_type=mdp_type, random_seed=random_seed, with_ligand=True, nnpot_active=nnpot_active, nnpot_modelfile_path=nnpot_modelfile_path, nnpot_input_group=nnpot_input_group, nnpot_model_name=nnpot_model_name, force_field=force_field)
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(file_path, file_content)
        status = "Production MD parameter file generated successfully."
    except Exception as exc:
        status = "Error generating production MD parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_prod_md_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                 parameter_file_name: str, run_input_file_name: str,
                                 max_warnings: int,
                                 force_field: str | None = None) -> tuple[list[str], str]:
    """Run grompp to build the production MD run input file."""
    try:
        max_warnings = _normalise_max_warnings(max_warnings)
        parameter_path, topology_path = _validate_grompp_inputs(
            working_directory_path, parameter_file_name,
            input_topology_file_name, force_field)
        compatibility_warning = _custom_force_field_warning(topology_path)
        cmd = [
            "gmx", "grompp",
            "-f", parameter_path,
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", topology_path,
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]
        checkpoint_path = get_matching_checkpoint_path(working_directory_path, input_file_name)
        if checkpoint_path is not None:
            cmd.extend(["-t", checkpoint_path])

        print(f"Running command: {' '.join(cmd)}")

        gromos_warning = run_grompp_with_gromos_warning_policy(
            cmd, working_directory_path, topology_path, max_warnings,
            runner=run_checked_command)
        continuation_warning = " ".join(
            warning for warning in (compatibility_warning, gromos_warning)
            if warning) or None
        if checkpoint_path is None:
            checkpoint_warning = ("No matching checkpoint was found; GROMACS can read "
                                  "velocities from the input structure, but coupling "
                                  "state cannot be carried over.")
            continuation_warning = " ".join(
                warning for warning in (continuation_warning, checkpoint_warning)
                if warning)
        status, color = _grompp_success(
            "Production MD run input file generated successfully.",
            max_warnings,
            continuation_warning,
        )
    except Exception as exc:
        status = "Error generating production MD run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), f"<span style='color:{color};'>" + status + "</span>"

def on_run_prod_md(working_directory_path: str, run_input_file_name: str, mpi_rank: int, omp_threads: int,
                   prod_md_nnpot_active: bool, use_gpu: bool,
                   process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start production MD, or stop the run that is already in progress."""
    # ---------- STOP ----------
    proc, job_key = clear_process_state_for_directory(process_state, working_directory_path)
    if proc is not None:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    proc = None
    job_key = None
    try:
        if prod_md_nnpot_active:
            mpi_rank = 1
        mpi_rank, omp_threads = _validate_mdrun_resources(
            mpi_rank, omp_threads)
        base_name = os.path.splitext(run_input_file_name)[0]
        job_key, claimed, active_proc = _claim_process_output(
            working_directory_path, base_name, process_state, "Production MD",
            f"See {base_name}.log for details.")
        if not claimed:
            if active_proc is not None:
                status = (f"{get_process_job_name(process_state)} is already running "
                          "for this output. This session is now attached to it; "
                          "click Stop to end it.")
                button = gr.update(value="Stop", variant="stop")
            else:
                status = "Production MD is already starting for this output."
                button = gr.update(value="Start", variant="primary")
            return (get_files_in_working_directory(working_directory_path),
                    f"<span style='color:orange;'>{status}</span>", process_state, button)
        resource_status = reserve_process_resources(
            job_key, mpi_rank, omp_threads, use_gpu)
        # -deffnm also changes mdrun's optional -cpi default, so omitting -cpi
        # would silently resume <base_name>.cpt whenever one already exists.
        # Point it at a file inside a private temporary directory that never
        # exists; after Popen returns the directory can disappear as well.
        with tempfile.TemporaryDirectory(
                prefix="gromacs_webui_fresh_checkpoint_") as checkpoint_guard:
            cmd = [
                "gmx", "mdrun",
                "-deffnm", base_name,
                "-cpi", os.path.join(checkpoint_guard, "absent.cpt"),
                "-ntmpi", str(mpi_rank),
                "-ntomp", str(omp_threads),
                "-v"
            ]
            if use_gpu and not prod_md_nnpot_active:
                if int(mpi_rank) == 1:
                    # A fully GPU-resident update is only supported with one rank.
                    # With domain decomposition, retain only the safe non-bonded
                    # offload and let GROMACS place the remaining tasks.
                    cmd.extend([
                        "-nb", "gpu",
                        "-pme", "gpu",
                        "-bonded", "gpu",
                        "-update", "gpu",
                        "-pin", "on",
                        "-dlb", "yes"
                    ])
                else:
                    cmd.extend(get_mdrun_hardware_options(True, mpi_rank))
            elif not use_gpu:
                # Not merely "do not ask for the GPU": every task defaults to auto,
                # which picks a detected GPU, so the CPU has to be named.
                cmd.extend(get_cpu_only_mdrun_options())

            print(f"Running command: {' '.join(cmd)}")

            proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True,
                                    start_new_session=True)
        _activate_process(proc, process_state, job_key, "Production MD",
                          working_directory_path, f"See {base_name}.log for details.")

        status = f"Production MD started. {resource_status}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        _clean_up_failed_process_start(job_key, proc, process_state)

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_continue_prod_md(working_directory_path: str, run_input_file_name: str, checkpoint_file_name: str,
                        mpi_rank: int, omp_threads: int, prod_md_nnpot_active: bool, use_gpu: bool,
                        process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Resume an interrupted production run, or stop the resumed process."""
    # ---------- STOP ----------
    proc, job_key = clear_process_state_for_directory(process_state, working_directory_path)
    if proc is not None:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    proc = None
    job_key = None
    try:
        if prod_md_nnpot_active:
            mpi_rank = 1
        mpi_rank, omp_threads = _validate_mdrun_resources(
            mpi_rank, omp_threads)
        run_input_file_name, checkpoint_file_name = require_matching_resume_files(
            working_directory_path, run_input_file_name, checkpoint_file_name)
        base_name = os.path.splitext(run_input_file_name)[0]
        # Initial and continuation runs share this key: both write the same
        # trajectory, log and checkpoint under ``-deffnm <base_name>``.
        job_key, claimed, active_proc = _claim_process_output(
            working_directory_path, base_name, process_state, "Production MD",
            f"See {base_name}.log for details.")
        if not claimed:
            if active_proc is not None:
                status = (f"{get_process_job_name(process_state)} is already running "
                          "for this output. This session is now attached to it; "
                          "click Stop to end it.")
                button = gr.update(value="Stop", variant="stop")
            else:
                status = "Production MD is already starting for this output."
                button = gr.update(value="Start", variant="primary")
            return (get_files_in_working_directory(working_directory_path),
                    f"<span style='color:orange;'>{status}</span>", process_state, button)
        resource_status = reserve_process_resources(
            job_key, mpi_rank, omp_threads, use_gpu)
        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-cpi", checkpoint_file_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-append",
            "-v"
        ]
        if use_gpu and not prod_md_nnpot_active:
            if int(mpi_rank) == 1:
                cmd.extend([
                    "-nb", "gpu",
                    "-pme", "gpu",
                    "-bonded", "gpu",
                    "-update", "gpu",
                    "-pin", "on",
                    "-dlb", "yes"
                ])
            else:
                cmd.extend(get_mdrun_hardware_options(True, mpi_rank))
        elif not use_gpu:
            # Not merely "do not ask for the GPU": every task defaults to auto,
            # which picks a detected GPU, so the CPU has to be named.
            cmd.extend(get_cpu_only_mdrun_options())

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True,
                                start_new_session=True)
        _activate_process(proc, process_state, job_key, "Production MD",
                          working_directory_path, f"See {base_name}.log for details.")

        status = f"Interrupted production MD resumed. {resource_status}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        _clean_up_failed_process_start(job_key, proc, process_state)

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")
    
def on_make_molecule_whole(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                           output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -pbc whole to repair molecules broken across the box edge."""
    try:
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".trajectory_stage_",
                dir=working_directory_path) as stage_directory:
            staged_output = os.path.join(stage_directory, output_traj_file_name)
            cmd = [
                "gmx", "trjconv",
                "-s", os.path.join(working_directory_path, run_input_file_name),
                "-f", os.path.join(working_directory_path, input_traj_file_name),
                "-o", staged_output,
                "-pbc", "whole"
            ]

            print(f"Running command: {' '.join(cmd)}")

            group_input = get_gmx_group_input(
                cmd, ["System"], working_directory_path)
            run_checked_command(
                cmd, cwd=working_directory_path, stdin_input=group_input)
            _publish_staged_files([
                (staged_output,
                 os.path.join(working_directory_path, output_traj_file_name)),
            ])
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_center_protein(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                      output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -pbc mol -center to keep the solute in the middle of the box."""
    try:
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".trajectory_stage_",
                dir=working_directory_path) as stage_directory:
            staged_output = os.path.join(stage_directory, output_traj_file_name)
            cmd = [
                "gmx", "trjconv",
                "-s", os.path.join(working_directory_path, run_input_file_name),
                "-f", os.path.join(working_directory_path, input_traj_file_name),
                "-o", staged_output,
                "-pbc", "mol",
                "-center"
            ]

            print(f"Running command: {' '.join(cmd)}")

            group_input = get_gmx_group_input(
                cmd, ["Protein", "System"], working_directory_path)
            run_checked_command(
                cmd, cwd=working_directory_path, stdin_input=group_input)
            _publish_staged_files([
                (staged_output,
                 os.path.join(working_directory_path, output_traj_file_name)),
            ])
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_fit_backbone(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                    output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -fit rot+trans to remove overall rotation and translation."""
    try:
        with reserve_working_directory_maintenance(working_directory_path), \
                tempfile.TemporaryDirectory(
                prefix=".trajectory_stage_",
                dir=working_directory_path) as stage_directory:
            staged_output = os.path.join(stage_directory, output_traj_file_name)
            cmd = [
                "gmx", "trjconv",
                "-s", os.path.join(working_directory_path, run_input_file_name),
                "-f", os.path.join(working_directory_path, input_traj_file_name),
                "-o", staged_output,
                "-fit", "rot+trans"
            ]

            print(f"Running command: {' '.join(cmd)}")

            group_input = get_gmx_group_input(
                cmd, ["Backbone", "System"], working_directory_path)
            run_checked_command(
                cmd, cwd=working_directory_path, stdin_input=group_input)
            _publish_staged_files([
                (staged_output,
                 os.path.join(working_directory_path, output_traj_file_name)),
            ])
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def _require_ligand(universe):
    """The ligand selection every complex analysis depends on.

    Uploads are normalised to LIG (see on_upload_ligand_structure_file), so an
    empty selection here means the structure came from somewhere else. Say that,
    rather than letting MDAnalysis raise about an empty AtomGroup.
    """
    ligand_selector = universe.select_atoms(f"resname {LIGAND_RESNAME}")
    if ligand_selector.n_atoms == 0:
        raise Exception(f"No residue named {LIGAND_RESNAME} in this structure, so the "
                        f"ligand cannot be located. Uploading the ligand through this "
                        f"tab renames it to {LIGAND_RESNAME} automatically.")

    return ligand_selector


def _minimum_distance_in_chunks(first_positions: np.ndarray,
                                second_positions: np.ndarray,
                                box: np.ndarray | None,
                                max_pairs: int = MAX_DISTANCE_ARRAY_PAIRS) -> float:
    """Return an all-pairs minimum without allocating an unbounded matrix."""
    first_count = len(first_positions)
    second_count = len(second_positions)
    if first_count == 0 or second_count == 0:
        raise ValueError("Both atom selections must contain at least one atom.")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, numbers.Integral) \
            or max_pairs < 1:
        raise ValueError("Distance pair budget must be a positive integer.")

    # Split both axes: even a single very large selection must never make one
    # distance_array call exceed the configured pair budget.
    second_chunk = min(second_count, int(max_pairs))
    first_chunk = max(1, int(max_pairs) // second_chunk)
    minimum = math.inf
    for first_start in range(0, first_count, first_chunk):
        first = first_positions[first_start:first_start + first_chunk]
        for second_start in range(0, second_count, second_chunk):
            second = second_positions[second_start:second_start + second_chunk]
            pairwise = distances.distance_array(first, second, box=box)
            minimum = min(minimum, float(pairwise.min()))
    return minimum

def on_analyze_rmsd(working_directory_path: str, structure_file_name: str,
                    input_traj_file_name: str,
                    run_input_file_name: str | None = None) -> tuple[Any, ...]:
    """Protein and ligand RMSD after fitting the protein backbone.

    Both series stay on one plot: they are two readings of the same measurement
    and are compared against each other.  The ligand is deliberately not fitted
    on its own: its series therefore retains motion relative to the protein.
    """
    universe = None
    try:
        if run_input_file_name:
            times_ns, rmsd_values = gromacs_backbone_fitted_rmsd(
                working_directory_path, run_input_file_name,
                input_traj_file_name, ["Protein", LIGAND_RESNAME],
                group_resolver=get_gmx_group_input,
                command_runner=run_checked_command)
            frame = pd.DataFrame({"Time (ns)": times_ns,
                                  "Protein RMSD (Å)": rmsd_values[:, 0],
                                  "Ligand RMSD (Å)": rmsd_values[:, 1]})
            figure = make_line_figure(
                frame, "Time (ns)", ylabel="RMSD (Å)", title="RMSD vs Time")
            status = "RMSD calculated successfully with TPR-aware PBC correction."
            return frame, figure, "<span style='color:green;'>" + status + "</span>"

        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        _require_ligand(universe)

        fitted_rmsd = rms.RMSD(
            universe,
            select="protein and backbone",
            groupselections=["protein", f"resname {LIGAND_RESNAME}"],
            ref_frame=0
        ).run()

        values = np.asarray(fitted_rmsd.results.rmsd, dtype=float)
        if (values.ndim != 2 or values.shape[1] < 5 or values.shape[0] == 0
                or not np.all(np.isfinite(values[:, [1, 3, 4]]))
                or np.any(np.diff(values[:, 1]) < 0)
                or np.any(values[:, 3:] < 0)):
            raise ValueError("MDAnalysis returned invalid RMSD or trajectory time values.")
        frame = pd.DataFrame({"Time (ns)": values[:, 1] / 1000,
                              # Column 2 is the backbone fit RMSD.  Subsequent
                              # columns are the requested groups after that one
                              # shared fit, without further superposition.
                              "Protein RMSD (Å)": values[:, 3],
                              "Ligand RMSD (Å)": values[:, 4]})
        figure = make_line_figure(frame, "Time (ns)", ylabel="RMSD (Å)", title="RMSD vs Time")
        status = ("RMSD calculated successfully. Warning: no TPR was supplied, "
                  "so molecular PBC/connectivity correction was unavailable.")
    except Exception as exc:
        status = "Error calculating RMSD!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"
    finally:
        if universe is not None:
            universe.trajectory.close()

    return frame, figure, "<span style='color:orange;'>" + status + "</span>"

def on_analyze_min_distance(working_directory_path: str, structure_file_name: str,
                            input_traj_file_name: str) -> tuple[Any, ...]:
    """Closest approach between any protein atom and any ligand atom, per frame.

    Complements the centre of mass distance: two molecules in contact can still
    have their centres far apart, so this is what tells you whether the ligand is
    actually touching the protein rather than merely near it.
    """
    universe = None
    try:
        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        protein_selector = universe.select_atoms("protein")
        if protein_selector.n_atoms == 0:
            raise Exception("No protein atoms found. Is this a protein-ligand complex?")
        ligand_selector = _require_ligand(universe)

        times_ns = []
        minimum_distances = []
        for timestep in universe.trajectory:
            # The box is passed so a ligand that has wrapped around the periodic
            # boundary still measures as close as it physically is. A structure
            # without box information gives None, which distance_array accepts.
            times_ns.append(timestep.time / 1000)
            minimum_distances.append(_minimum_distance_in_chunks(
                protein_selector.positions, ligand_selector.positions,
                universe.dimensions))

        frame = pd.DataFrame({"Time (ns)": times_ns,
                              "Minimum distance (Å)": minimum_distances})
        figure = make_line_figure(frame, "Time (ns)", ylabel="Minimum distance (Å)",
                                  title="Protein-ligand minimum distance", mean_line=True)
        status = "Minimum distance calculated successfully."
    except Exception as exc:
        status = "Error calculating minimum distance!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"
    finally:
        if universe is not None:
            universe.trajectory.close()

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_com_distance(working_directory_path: str, structure_file_name: str,
                            input_traj_file_name: str,
                            run_input_file_name: str | None = None) -> tuple[Any, ...]:
    """Distance between the protein and ligand centres of mass, frame by frame.

    A TPR contains the molecular connectivity that is needed to make each
    molecule whole before calculating its centre.  The UI therefore takes the
    topology-aware GROMACS path.  Keep the coordinate-only MDAnalysis path for
    backwards-compatible direct use with imported trajectories that have no
    TPR; its atom-order unwrapping is necessarily only a best effort.
    """
    universe = None
    temporary_output_path = None
    try:
        if run_input_file_name:
            descriptor, temporary_output_path = tempfile.mkstemp(
                prefix=".com_distance_", suffix=".xvg",
                dir=working_directory_path)
            os.close(descriptor)
            # gmx refuses to replace an existing file without creating a
            # backup.  Reserve a collision-free name, then release it for gmx.
            os.unlink(temporary_output_path)
            temporary_output_name = os.path.basename(temporary_output_path)

            cmd = [
                "gmx", "distance",
                "-s", run_input_file_name,
                "-f", input_traj_file_name,
                "-select", 'com of group "Protein" plus com of resname LIG',
                "-oall", temporary_output_name,
                "-tu", "ns",
                "-xvg", "none",
                "-rmpbc", "yes",
                "-pbc", "yes",
            ]
            run_checked_command(cmd, cwd=working_directory_path, stdin_input="")
            raw = read_xvg(temporary_output_path)["frame"]
            if raw.shape[1] != 2:
                raise ValueError(
                    "gmx distance returned an unexpected number of columns.")

            values = raw.to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("gmx distance returned non-finite values.")
            frame = pd.DataFrame({
                "Time (ns)": values[:, 0],
                # GROMACS length output is nm; the rest of this analysis panel
                # consistently presents structural distances in angstroms.
                "Center of mass distance (Å)": values[:, 1] * 10.0,
            })
            figure = make_line_figure(
                frame, "Time (ns)", ylabel="Center of mass distance (Å)",
                title="Protein-ligand centre of mass distance")
            status = "Center of mass distance calculated successfully."
            return frame, figure, "<span style='color:green;'>" + status + "</span>"

        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        protein_selector = universe.select_atoms("protein")
        if protein_selector.n_atoms == 0:
            raise Exception("No protein atoms found. Is this a protein-ligand complex?")
        ligand_selector = _require_ligand(universe)

        # The time axis is built here rather than borrowed from an RMSD result, so
        # this analysis stands on its own now that it has its own button.
        # Not named "distances": that is the MDAnalysis module this file imports,
        # and shadowing it would break any later use of distance_array here.
        times_ns = []
        com_distances = []
        for timestep in universe.trajectory:
            times_ns.append(timestep.time / 1000)
            # distance_array applies the triclinic minimum-image convention when
            # this frame carries unit-cell dimensions, and behaves like ordinary
            # Euclidean distance when it does not.
            # Build each COM from molecule-contiguous coordinates first. An
            # ordinary COM can land near the box centre when a molecule straddles
            # a boundary, even though the molecule itself is nowhere near it.
            protein_com = periodic_center_of_mass(
                protein_selector, timestep.dimensions)[None, :]
            ligand_com = periodic_center_of_mass(
                ligand_selector, timestep.dimensions)[None, :]
            com_distances.append(float(distances.distance_array(
                protein_com, ligand_com, box=timestep.dimensions)[0, 0]))

        frame = pd.DataFrame({"Time (ns)": times_ns,
                              "Center of mass distance (Å)": com_distances})
        figure = make_line_figure(frame, "Time (ns)", ylabel="Center of mass distance (Å)",
                                  title="Protein-ligand centre of mass distance")
        status = "Center of mass distance calculated successfully."
    except Exception as exc:
        status = "Error calculating center of mass distance!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"
    finally:
        if universe is not None:
            universe.trajectory.close()
        if temporary_output_path and os.path.exists(temporary_output_path):
            os.unlink(temporary_output_path)

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_rmsf(working_directory_path: str, structure_file_name: str,
                    input_traj_file_name: str,
                    run_input_file_name: str | None = None) -> tuple[Any, ...]:
    """Backbone-aligned C-alpha fluctuation over the whole trajectory."""
    universe = None
    try:
        if run_input_file_name:
            residue_indices, residue_labels, ca_rmsf = \
                gromacs_topology_aware_ca_rmsf(
                    working_directory_path, run_input_file_name,
                    input_traj_file_name, structure_file_name,
                    group_resolver=get_gmx_group_input,
                    command_runner=run_checked_command)
            status = "RMSF calculated successfully with TPR-aware PBC correction."
            color = "green"
        else:
            universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                    os.path.join(working_directory_path, input_traj_file_name))
            residue_indices, residue_labels, ca_rmsf = backbone_aligned_ca_rmsf(universe)
            status = ("RMSF calculated successfully. Warning: no TPR was supplied, "
                      "so molecular PBC/connectivity correction was unavailable.")
            color = "orange"

        ca_rmsf = np.asarray(ca_rmsf, dtype=float)
        if (ca_rmsf.ndim != 1 or len(ca_rmsf) != len(residue_indices)
                or not np.all(np.isfinite(ca_rmsf)) or np.any(ca_rmsf < 0)):
            raise ValueError("Calculated C-alpha RMSF values are invalid.")

        frame = pd.DataFrame({"Residue Index": residue_indices,
                              "Residue": residue_labels,
                              "Cα RMSF (Å)": ca_rmsf})
        figure = make_line_figure(frame, "Residue Index",
                                  y_columns=["Cα RMSF (Å)"], ylabel="RMSF (Å)",
                                  title="Cα RMSF per Residue", mean_line=True)
    except Exception as exc:
        status = "Error calculating RMSF!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"
    finally:
        if universe is not None:
            universe.trajectory.close()

    return frame, figure, f"<span style='color:{color};'>" + status + "</span>"

def _selection_error(exc: Exception, run_input_file_name: str,
                     working_directory_path: str) -> str:
    """A gmx failure message, with the structure's own residue groups appended
    when the cause was a selection that matched nothing."""
    message = str(exc)
    if "never matches any atoms" not in message and "Invalid selection" not in message:
        return message

    hint = describe_selection_candidates(run_input_file_name, working_directory_path)
    return f"{message}\n\n{hint}" if hint else message

def _require_selected_files(**file_names: Any) -> None:
    """Fail with the empty dropdown's name rather than a TypeError deep in argv.

    A file dropdown holds None until the job directory contains a file of that
    kind, and None reaching the command line raises "sequence item 3: expected
    str instance" from ' '.join, which says nothing useful.
    """
    missing = [label for label, value in file_names.items() if not value]
    if missing:
        raise Exception("Select a file for: " + ", ".join(missing) + ".")

def on_analyze_sasa(working_directory_path: str, run_input_file_name: str,
                    input_traj_file_name: str, surface_selection: str, output_selection: str,
                    probe_radius: float, sasa_file_name: str,
                    sasa_residue_file_name: str) -> Any:
    """Solvent accessible surface area over time, and averaged per residue.

    A generator, so the command being run reaches the status markdown before it
    blocks rather than only afterwards: gmx sasa over a long trajectory can take
    minutes, and an unchanging page looks like nothing happened.

    stdin is closed for every gmx analysis here. These tools fall back to an
    interactive group prompt when a selection option is missing, and with a
    blocking stdin that wedges the worker thread indefinitely (measured: no
    -surface plus a live stdin hangs forever, stdin closed fails in a second).
    """
    try:
        _require_selected_files(**{"Run Input File Name": run_input_file_name,
                                   "Input Trajectory File Name": input_traj_file_name})
    except Exception as exc:
        yield get_files_in_working_directory(working_directory_path), None, None, None, None, \
            "<span style='color:red;'>Error calculating SASA!\n" + str(exc) + "</span>"
        return

    cmd = [
        "gmx", "sasa",
        "-s", run_input_file_name,
        "-f", input_traj_file_name,
        "-o", sasa_file_name,
        "-or", sasa_residue_file_name,
        "-surface", surface_selection,
        "-probe", str(probe_radius),
        "-tu", "ns"
    ]
    if output_selection and output_selection.strip():
        cmd.extend(["-output", output_selection])

    print(f"Running command (in {working_directory_path}): {' '.join(cmd)}")
    yield get_files_in_working_directory(working_directory_path), None, None, None, None, \
        format_running_status(cmd)

    try:
        run_checked_command(cmd, cwd=working_directory_path, stdin_input="")

        area = read_xvg(os.path.join(working_directory_path, sasa_file_name))
        # The x column name comes from the file: -tu rewrites the axis label, so
        # hardcoding "Time (ns)" would break the moment the unit changes.
        area_figure = make_line_figure(area["frame"], ylabel=area["ylabel"],
                                       title=area["title"] or "Solvent accessible surface area")

        residue = read_xvg(os.path.join(working_directory_path, sasa_residue_file_name))
        # gmx writes an average and a standard deviation per output group; plot the
        # average of the surface group and leave the rest to the exported table.
        residue_figure = make_line_figure(residue["frame"],
                                          y_columns=[residue["frame"].columns[1]],
                                          ylabel=residue["ylabel"],
                                          title=residue["title"] or "Area per residue")

        status = "SASA calculated successfully."
    except Exception as exc:
        status = "Error calculating SASA!\n" + _selection_error(
            exc, run_input_file_name, working_directory_path)
        yield get_files_in_working_directory(working_directory_path), None, None, None, None, \
            "<span style='color:red;'>" + status + "</span>"
        return

    yield get_files_in_working_directory(working_directory_path), area["frame"], area_figure, \
        residue["frame"], residue_figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_gyrate(working_directory_path: str, run_input_file_name: str,
                      input_traj_file_name: str, gyrate_selection: str, weighting_mode: str,
                      gyrate_file_name: str) -> Any:
    """Radius of gyration over time, total and about each axis."""
    try:
        _require_selected_files(**{"Run Input File Name": run_input_file_name,
                                   "Input Trajectory File Name": input_traj_file_name})
    except Exception as exc:
        yield get_files_in_working_directory(working_directory_path), None, None, \
            "<span style='color:red;'>Error calculating radius of gyration!\n" + str(exc) + "</span>"
        return

    cmd = [
        "gmx", "gyrate",
        "-s", run_input_file_name,
        "-f", input_traj_file_name,
        "-o", gyrate_file_name,
        "-sel", gyrate_selection,
        "-mode", weighting_mode,
        "-tu", "ns"
    ]

    print(f"Running command (in {working_directory_path}): {' '.join(cmd)}")
    yield get_files_in_working_directory(working_directory_path), None, None, \
        format_running_status(cmd)

    try:
        run_checked_command(cmd, cwd=working_directory_path, stdin_input="")

        gyration = read_xvg(os.path.join(working_directory_path, gyrate_file_name))
        figure = make_line_figure(gyration["frame"], ylabel=gyration["ylabel"],
                                  title=gyration["title"] or "Radius of gyration")

        status = "Radius of gyration calculated successfully."
    except Exception as exc:
        status = "Error calculating radius of gyration!\n" + _selection_error(
            exc, run_input_file_name, working_directory_path)
        yield get_files_in_working_directory(working_directory_path), None, None, \
            "<span style='color:red;'>" + status + "</span>"
        return

    yield get_files_in_working_directory(working_directory_path), gyration["frame"], figure, \
        "<span style='color:green;'>" + status + "</span>"

def on_run_pca(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
               pca_selection: str, first_eigenvector: int, second_eigenvector: int,
               pca_index_file_name: str, pca_eigenvector_file_name: str,
               pca_eigenvalue_file_name: str,
               pca_projection_file_name: str) -> Any:
    """Principal component analysis of the trajectory, via gmx covar and anaeig.

    A generator: three commands run back to back and covar is the slow one, so
    each is announced in the status markdown before it blocks.

    covar and anaeig are legacy tools that ask which group to fit and which to
    analyse. Rather than answering with a group number - which shifts with the
    force field and the contents of the system - a one-group index file is built
    first with gmx select, which leaves them nothing to ask about. Using the same
    index for both guarantees the fit and analysis groups match, which is what
    PCA wants, and that anaeig sees the same atom count the eigenvectors were
    built from.
    """
    files = get_files_in_working_directory(working_directory_path)
    try:
        _require_selected_files(**{"Run Input File Name": run_input_file_name,
                                   "Input Trajectory File Name": input_traj_file_name})
        # Checked before gmx runs: covar on a production trajectory takes minutes,
        # and failing afterwards would waste all of it and overwrite the outputs.
        first = int(first_eigenvector)
        second = int(second_eigenvector)
        if second <= first:
            raise Exception("The second eigenvector must be higher than the first.")

        select_cmd = [
            "gmx", "select",
            "-s", run_input_file_name,
            "-select", pca_selection,
            "-on", pca_index_file_name
        ]
        print(f"Running command (in {working_directory_path}): {' '.join(select_cmd)}")
        yield files, None, None, None, None, format_running_status(select_cmd, "Step 1 of 3")
        run_checked_command(select_cmd, cwd=working_directory_path, stdin_input="")

        covar_cmd = [
            "gmx", "covar",
            "-s", run_input_file_name,
            "-f", input_traj_file_name,
            "-n", pca_index_file_name,
            "-o", pca_eigenvalue_file_name,
            "-v", pca_eigenvector_file_name,
            "-av", "pca_average.pdb",
            "-l", "pca_covar.log",
            "-xvg", "xmgrace"
        ]
        print(f"Running command (in {working_directory_path}): {' '.join(covar_cmd)}")
        yield files, None, None, None, None, format_running_status(covar_cmd, "Step 2 of 3")
        run_checked_command(covar_cmd, cwd=working_directory_path, stdin_input="")

        anaeig_cmd = [
            "gmx", "anaeig",
            "-s", run_input_file_name,
            "-f", input_traj_file_name,
            "-n", pca_index_file_name,
            "-v", pca_eigenvector_file_name,
            "-eig", pca_eigenvalue_file_name,
            "-first", str(first),
            "-last", str(second),
            "-2d", pca_projection_file_name,
            "-xvg", "xmgrace"
        ]
        print(f"Running command (in {working_directory_path}): {' '.join(anaeig_cmd)}")
        yield files, None, None, None, None, format_running_status(anaeig_cmd, "Step 3 of 3")
        run_checked_command(anaeig_cmd, cwd=working_directory_path, stdin_input="")

        eigenvalues = read_xvg(os.path.join(working_directory_path, pca_eigenvalue_file_name))
        eigenvalue_figure = make_scree_figure(eigenvalues["frame"],
                                              title="Eigenvalues and cumulative variance")

        projection = read_xvg(os.path.join(working_directory_path, pca_projection_file_name))
        projection_figure = make_scatter_figure(
            projection["frame"], xlabel=projection["xlabel"], ylabel=projection["ylabel"],
            title=f"Projection on eigenvectors {first} and {second}")

        status = "PCA completed successfully."
    except Exception as exc:
        status = "Error running PCA!\n" + _selection_error(
            exc, run_input_file_name, working_directory_path)
        yield get_files_in_working_directory(working_directory_path), None, None, None, None, \
            "<span style='color:red;'>" + status + "</span>"
        return

    yield get_files_in_working_directory(working_directory_path), eigenvalues["frame"], \
        eigenvalue_figure, projection["frame"], projection_figure, \
        "<span style='color:green;'>" + status + "</span>"

def on_analyze_free_energy_landscape(working_directory_path: str, projection_file_name: str,
                                     temperature: float,
                                     bin_count: int) -> tuple[Any, ...]:
    """Gibbs free energy landscape over the two principal components.

    Reads the projection back off disk rather than taking it through gr.State, so
    the landscape can be recomputed at a different temperature or resolution
    without rerunning the PCA, and survives a page reload.
    """
    try:
        projection_file_path = os.path.join(working_directory_path, projection_file_name)
        if not os.path.exists(projection_file_path):
            raise Exception(f"{projection_file_name} was not found. Run the PCA first.")

        projection = read_xvg(projection_file_path)
        if len(projection["frame"].columns) < 2:
            raise Exception(f"{projection_file_name} has only one column, so it holds no "
                            f"2D projection. Rerun the PCA to write it.")

        first_component = projection["frame"].iloc[:, 0]
        second_component = projection["frame"].iloc[:, 1]
        x_centres, y_centres, probability, free_energy = compute_free_energy_landscape(
            first_component, second_component, bin_count=int(bin_count),
            temperature=float(temperature))

        figure = make_landscape_figure(x_centres, y_centres, free_energy,
                                       xlabel=projection["xlabel"] or "PC1",
                                       ylabel=projection["ylabel"] or "PC2",
                                       title=f"Free energy landscape at {float(temperature):g} K")

        # Long form, one row per bin, so the surface exports as a plain table.
        x_grid, y_grid = np.meshgrid(x_centres, y_centres, indexing="ij")
        frame = pd.DataFrame({
            projection["xlabel"] or "PC1": x_grid.ravel(),
            projection["ylabel"] or "PC2": y_grid.ravel(),
            "Probability": probability.ravel(),
            "ΔG (kJ/mol)": free_energy.ravel(),
        })

        status = "Free energy landscape calculated successfully."
    except Exception as exc:
        status = "Error calculating free energy landscape!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

MMPBSA_SUBDIRECTORY: str = "mmpbsa"
MMPBSA_RESULTS_FILE_NAME: str = "FINAL_RESULTS_MMPBSA.dat"
MMPBSA_LOG_FILE_NAME: str = "mmpbsa_run.log"
# -eo gives every energy term per frame, which is what the binding energy
# histogram is built from; -do and -deo are the per-residue decomposition.
MMPBSA_PER_FRAME_FILE_NAME: str = "FINAL_RESULTS_MMPBSA.csv"
MMPBSA_DECOMP_FILE_NAME: str = "FINAL_DECOMP_MMPBSA.dat"
MMPBSA_DECOMP_PER_FRAME_FILE_NAME: str = "FINAL_DECOMP_MMPBSA.csv"
MMPBSA_RESULT_ARTIFACTS: tuple[str, ...] = (
    MMPBSA_RESULTS_FILE_NAME,
    MMPBSA_PER_FRAME_FILE_NAME,
    MMPBSA_DECOMP_FILE_NAME,
    MMPBSA_DECOMP_PER_FRAME_FILE_NAME,
    MMPBSA_LOG_FILE_NAME,
)
# How many residues the contribution chart shows; the exported table keeps all.
MMPBSA_DECOMPOSITION_RESIDUES_SHOWN: int = 15


def _legacy_mmpbsa_directory(working_directory_path: str) -> str:
    """Resolve the historical result directory without allowing a symlink escape."""
    job_directory = Path(working_directory_path).resolve()
    legacy_directory = (job_directory / MMPBSA_SUBDIRECTORY).resolve()
    if legacy_directory.parent != job_directory:
        raise ValueError("Invalid legacy MM-PBSA directory: it must stay inside the job.")
    return str(legacy_directory)


def _restore_legacy_mmpbsa_results(working_directory_path: str,
                                    legacy_directory_path: str,
                                    selected_results_file_name: str,
                                    overwrite: bool = True) -> list[str]:
    """Copy a legacy run's complete result set back into the job directory.

    Older versions ran gmx_MMPBSA in ``mmpbsa/``.  Copying only its summary made
    the optional plots work once, then disappear on the next load because the
    newly visible summary was beside none of its companion CSV files.
    """
    expected_legacy_directory = _legacy_mmpbsa_directory(working_directory_path)
    if Path(legacy_directory_path).resolve() != Path(expected_legacy_directory):
        raise ValueError("Invalid legacy MM-PBSA result path.")

    restored: list[str] = []
    artifact_names = dict.fromkeys((selected_results_file_name, *MMPBSA_RESULT_ARTIFACTS))
    for file_name in artifact_names:
        source_path = os.path.join(legacy_directory_path, file_name)
        if not os.path.isfile(source_path):
            continue
        destination_path = os.path.join(working_directory_path, file_name)
        if not overwrite and os.path.exists(destination_path):
            continue
        shutil.copy2(source_path, destination_path)
        restored.append(file_name)
    return restored

def _whole_number(value: Any, label: str, minimum: int = 0) -> int:
    """Read a frame number out of a textbox, or say which box is wrong.

    The frame range is typed rather than dragged, because a production
    trajectory holds more frames than any slider range would guess.
    """
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise Exception(f"{label} must be a whole number, not '{value}'.") from None

    if number < minimum:
        raise Exception(f"{label} must be {minimum} or greater.")

    return number

def on_generate_mmpbsa_input_file(working_directory_path: str, mmpbsa_input_file_name: str,
                                  start_frame: str, end_frame: str, interval: int,
                                  salt_concentration: float, temperature: float,
                                  methods: Sequence[str], use_decomposition: bool,
                                  decomposition_scheme: int,
                                  print_residues: str) -> tuple[list[str], str]:
    """Write the &general/&gb/&pb/&decomp namelists gmx_MMPBSA reads."""
    try:
        first = _whole_number(start_frame, "Start Frame", minimum=1)
        last = _whole_number(end_frame, "End Frame", minimum=0)
        if last and last < first:
            raise Exception(f"End Frame ({last}) is before Start Frame ({first}). "
                            f"Use 0 to run to the end of the trajectory.")

        file_content = get_default_mmpbsa_input_file_content(
            start_frame=first, end_frame=last, interval=interval,
            salt_concentration=salt_concentration, temperature=temperature,
            use_gb="MM-GBSA" in methods, use_pb="MM-PBSA" in methods,
            use_decomposition=bool(use_decomposition),
            decomposition_scheme=decomposition_scheme,
            print_residues=print_residues)

        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_text_file(
                os.path.join(working_directory_path, mmpbsa_input_file_name),
                file_content)
        status = "MM-PBSA input file generated successfully."
    except Exception as exc:
        status = "Error generating MM-PBSA input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"

    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def _build_mmpbsa_index(working_directory_path: str, run_input_file_name: str,
                        receptor_selection: str, ligand_selection: str,
                        mmpbsa_index_file_name: str) -> None:
    """Write a two-group index: receptor first, ligand second.

    gmx select writes the groups in the order they are given, so -cg 0 1 always
    means receptor then ligand and no group number has to be guessed. Both are
    checked here because an empty ligand group only surfaces as a gmx_MMPBSA
    failure much later, after the expensive part has already run.
    """
    # One -select holding both selections separated by ";". Passing -select twice
    # is rejected outright ("Option specified multiple times"); the semicolon form
    # writes one group per selection, in the order given, so the receptor is
    # always group 0 and the ligand group 1.
    cmd = [
        "gmx", "select",
        "-s", run_input_file_name,
        "-select", f"{receptor_selection}; {ligand_selection}",
        "-on", mmpbsa_index_file_name
    ]
    print(f"Running command (in {working_directory_path}): {' '.join(cmd)}")
    run_checked_command(cmd, cwd=working_directory_path, stdin_input="")

    with open(os.path.join(working_directory_path, mmpbsa_index_file_name)) as handle:
        index_content = handle.read()

    groups = [block for block in index_content.split("[") if block.strip()]
    if len(groups) != 2:
        raise Exception(f"Expected a receptor group and a ligand group in "
                        f"{mmpbsa_index_file_name}, found {len(groups)}.")
    for name, block in zip((receptor_selection, ligand_selection), groups):
        if not block.split("]", 1)[1].split():
            raise Exception(f"The selection '{name}' matched no atoms.")

def on_run_mmpbsa(working_directory_path: str, run_input_file_name: str,
                  input_traj_file_name: str, input_topology_file_name: str,
                  mmpbsa_input_file_name: str, mmpbsa_index_file_name: str,
                  receptor_selection: str, ligand_selection: str, mmpbsa_processes: int,
                  process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start an MM-PBSA/MM-GBSA run, or stop the one already in progress.

    gmx_MMPBSA is run as an external command rather than imported: it pins older
    numpy, pandas and AmberTools than this application uses, so it lives in its
    own environment and the two dependency sets never meet.
    """
    # ---------- STOP ----------
    proc, job_key = clear_process_state_for_directory(process_state, working_directory_path)
    if proc is not None:
        # Outside the lock: waiting on the shutdown must not block the timer that
        # polls this state.
        stop_process_gracefully(proc)
        release_process_job(job_key, proc)

        status = "MM-PBSA stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    proc = None
    job_key = None
    try:
        mmpbsa_processes = _validate_positive_integer_resource(
            mmpbsa_processes, "MM-PBSA processes", get_default_cpu_count())
        executable = get_gmx_mmpbsa_executable()
        if executable is None:
            # "or" guards the case where the two helpers disagree: raising
            # Exception(None) would show the user the word "None".
            raise Exception(get_gmx_mmpbsa_unavailable_reason()
                            or "gmx_MMPBSA was not found. See the Readme for how to install it.")

        job_key, claimed, active_proc = _claim_process_output(
            working_directory_path, MMPBSA_RESULTS_FILE_NAME, process_state,
            "MM-PBSA", f"See {MMPBSA_LOG_FILE_NAME} for details.")
        if not claimed:
            if active_proc is not None:
                status = (f"{get_process_job_name(process_state)} is already running "
                          "in this working directory. This session is now attached "
                          "to it; click Stop to end it.")
                button = gr.update(value="Stop", variant="stop")
            else:
                status = "MM-PBSA is already starting in this working directory."
                button = gr.update(value="Start", variant="primary")
            return (get_files_in_working_directory(working_directory_path),
                    f"<span style='color:orange;'>{status}</span>", process_state, button)

        resource_status = reserve_process_resources(
            job_key, mmpbsa_processes, 1, False,
            request_label="MM-PBSA processes",
            reduction_hint="reduce MM-PBSA processes")

        _build_mmpbsa_index(working_directory_path, run_input_file_name, receptor_selection,
                            ligand_selection, mmpbsa_index_file_name)

        # gmx_MMPBSA scatters dozens of _GMXMMPBSA_* scratch files, so it runs in a
        # subdirectory. The file listing skips directories, so they stay out of the
        # file table; only the results are copied back out.
        # Run in the job directory itself, not a scratch subdirectory. A topology
        # is not self-contained: it #includes ligand_GMX.itp, posre.itp and
        # whatever else sits beside it, resolved relative to the working
        # directory. Copying the named inputs somewhere else leaves those behind
        # and the run dies in the preprocessor. The scratch files gmx_MMPBSA
        # leaves are hidden from the file listing instead.
        cmd = [
            executable, "-O", "-nogui",
            "-i", mmpbsa_input_file_name,
            "-cs", run_input_file_name,
            "-ct", input_traj_file_name,
            "-ci", mmpbsa_index_file_name,
            "-cg", "0", "1",
            "-cp", input_topology_file_name,
            "-o", MMPBSA_RESULTS_FILE_NAME,
            "-eo", MMPBSA_PER_FRAME_FILE_NAME,
            "-do", MMPBSA_DECOMP_FILE_NAME,
            "-deo", MMPBSA_DECOMP_PER_FRAME_FILE_NAME
        ]
        if int(mmpbsa_processes) > 1:
            cmd = [get_mpirun_beside(executable), "-np", str(int(mmpbsa_processes))] + cmd + ["MPI"]

        print(f"Running command (in {working_directory_path}): {' '.join(cmd)}")

        # Everything the run prints goes to a log in the job directory, so a
        # failure can be read in the text viewer instead of only in the terminal
        # the server happens to be attached to. The handle is closed straight
        # away; the child keeps its own descriptor.
        log_file_path = os.path.join(working_directory_path, MMPBSA_LOG_FILE_NAME)
        with open(log_file_path, "w") as log_file:
            # stdin is closed: gmx_MMPBSA prompts before overwriting in some paths,
            # and an inherited stdin would wait on an answer nobody sees.
            proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True,
                                    stdin=subprocess.DEVNULL, stdout=log_file,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env=get_gmx_mmpbsa_environment(executable))
        _activate_process(proc, process_state, job_key, "MM-PBSA",
                          working_directory_path,
                          f"See {MMPBSA_LOG_FILE_NAME} for details.")

        status = (f"MM-PBSA started. This can take a long time; load the results when "
                  f"the button returns to Start. Progress and any error are written to "
                  f"{MMPBSA_LOG_FILE_NAME}, which opens in the text viewer. "
                  f"{resource_status}")

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        _clean_up_failed_process_start(job_key, proc, process_state)

        status = "Error starting MM-PBSA:<br>" + _selection_error(
            exc, run_input_file_name, working_directory_path).replace("\n", "<br>")

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_load_mmpbsa_results(working_directory_path: str, mmpbsa_results_file_name: str,
                           structure_file_name: str, input_traj_file_name: str,
                           mmpbsa_input_file_name: str) -> tuple[Any, ...]:
    """Read the finished run's energy decomposition into a table and a bar chart.

    Separate from the run button because the run is asynchronous: nothing can be
    returned at the moment it is launched.
    """
    try:
        if mmpbsa_results_file_name != MMPBSA_RESULTS_FILE_NAME:
            raise ValueError(
                f"Only {MMPBSA_RESULTS_FILE_NAME} can be loaded because its "
                "companion per-frame and decomposition file names are fixed."
            )
        legacy_directory_path: str | None = None
        restore_only_missing = False
        results_file_path = os.path.join(working_directory_path, mmpbsa_results_file_name)
        legacy_path = os.path.join(
            _legacy_mmpbsa_directory(working_directory_path), mmpbsa_results_file_name)
        if not os.path.exists(results_file_path):
            # Runs started before the move out of the scratch subdirectory left
            # their results there, so those stay readable.
            if os.path.exists(legacy_path):
                results_file_path = legacy_path
                legacy_directory_path = os.path.dirname(legacy_path)
            else:
                raise Exception(f"{mmpbsa_results_file_name} was not found. Has the run "
                                f"finished? {MMPBSA_LOG_FILE_NAME} shows how far it got.")
        elif os.path.isfile(legacy_path) and filecmp.cmp(
                results_file_path, legacy_path, shallow=False):
            # A release between the old and new layouts copied only the summary.
            # Matching summaries identify that interrupted migration without
            # mixing an unrelated old legacy run into a newer root-level run.
            legacy_directory_path = os.path.dirname(legacy_path)
            restore_only_missing = True

        frame = parse_mmpbsa_results(results_file_path)
        # A run that asked for both GB and PB reports every term twice. Say which
        # is which on the axis, rather than drawing pairs of identical labels.
        label_column, methods = "Term", sorted(frame["Method"].unique())
        if len(methods) > 1:
            label_column = "Term (method)"
            frame = frame.assign(**{label_column: frame["Term"] + " (" + frame["Method"] + ")"})

        # Error bars use the plain per-frame SD rather than SD(Prop.): the
        # propagated one describes the components, not the spread of the delta.
        figure = make_bar_figure(frame, label_column, "Average (kcal/mol)", "SD",
                                 ylabel="ΔG (kcal/mol)",
                                 title="MM-PBSA energy decomposition: " + ", ".join(methods))

        migration_note = ""
        if legacy_directory_path is not None:
            restored = _restore_legacy_mmpbsa_results(
                working_directory_path, legacy_directory_path, mmpbsa_results_file_name,
                overwrite=not restore_only_missing)
            results_directory_path = working_directory_path
            if restored:
                migration_note = (f" Restored {len(restored)} legacy MM-PBSA result "
                                  f"file{'s' if len(restored) != 1 else ''} to the job directory.")
        else:
            results_directory_path = os.path.dirname(results_file_path)
        # The per-frame and per-residue files sit beside the summary and are only
        # written when the run asked for them, so each is optional.
        # Each extra is loaded on its own: a malformed companion file must cost
        # only its own panel, not the summary table and chart that already parsed.
        histogram_figure, missing = _load_panel(
            _load_binding_energy_histogram, 1, results_directory_path)
        series_figure, series_note = _load_panel(
            _load_binding_energy_series, 1, results_directory_path, working_directory_path,
            structure_file_name, input_traj_file_name, mmpbsa_input_file_name)
        decomposition, decomposition_figure, decomposition_missing = _load_panel(
            _load_residue_decomposition, 2, results_directory_path)

        status = "MM-PBSA results loaded successfully." + migration_note
        for note in (missing, series_note, decomposition_missing):
            if note:
                status += " " + note
    except Exception as exc:
        status = "Error loading MM-PBSA results!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), None, None, None, \
            None, None, None, "<span style='color:red;'>" + status + "</span>"

    return get_files_in_working_directory(working_directory_path), frame, figure, \
        series_figure, histogram_figure, decomposition, decomposition_figure, \
        "<span style='color:green;'>" + status + "</span>"

def _load_panel(loader: Any, result_count: int, *arguments: Any) -> tuple[Any, ...]:
    """Run one optional results panel, turning a failure into a note.

    The panels are independent: the residue decomposition failing to parse should
    not take the summary table, the histogram and the time series down with it,
    which is what a single try/except around all of them did.
    """
    try:
        return loader(*arguments)
    except Exception as exc:
        return (None,) * result_count + (f"{type(exc).__name__} while reading the "
                                         f"{loader.__name__.removeprefix('_load_')} "
                                         f"panel: {exc}",)

def _load_binding_energy_histogram(results_directory_path: str) -> tuple[Any, str]:
    """The spread of the binding energy over the frames, if -eo was written."""
    per_frame_path = os.path.join(results_directory_path, MMPBSA_PER_FRAME_FILE_NAME)
    if not os.path.exists(per_frame_path):
        return None, (f"No {MMPBSA_PER_FRAME_FILE_NAME}, so there is no per-frame "
                      f"distribution to plot.")

    per_frame = parse_mmpbsa_per_frame(per_frame_path)
    methods = list(per_frame["Method"].drop_duplicates())
    if len(methods) == 1:
        figure = make_histogram_figure(per_frame["TOTAL"], bins=30,
                                       xlabel="ΔG binding (kcal/mol)",
                                       title=f"Binding energy over {len(per_frame)} frames")
    else:
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 6))
        axes = figure.subplots()
        colours = [f"C{index}" for index in range(len(methods))]
        for colour, method in zip(colours, methods):
            values = per_frame.loc[per_frame["Method"] == method, "TOTAL"].to_numpy()
            axes.hist(values, bins=30, alpha=0.5, color=colour,
                      edgecolor="white", label=f"{method} ({len(values)} frames)")
            axes.axvline(values.mean(), color=colour, linestyle="--",
                         label=f"{method} mean {values.mean():.2f}")
        axes.set_xlabel("ΔG binding (kcal/mol)")
        axes.set_ylabel("Frames")
        axes.set_title("Binding energy distributions: " + ", ".join(methods))
        axes.legend()
        figure.tight_layout()
    return figure, ""

def _load_residue_decomposition(results_directory_path: str) -> tuple[Any, Any, str]:
    """Per-residue contributions, if the run enabled decomposition."""
    decomposition_path = os.path.join(results_directory_path,
                                      MMPBSA_DECOMP_PER_FRAME_FILE_NAME)
    if not os.path.exists(decomposition_path):
        return None, None, (f"No {MMPBSA_DECOMP_PER_FRAME_FILE_NAME}: tick "
                            f"'Per-residue decomposition' before running to get "
                            f"residue contributions.")

    decomposition = parse_mmpbsa_decomposition(decomposition_path)
    if "TOTAL" not in decomposition:
        # The parser tolerates a file without a TOTAL column, so the chart has to
        # as well; the table is still worth showing and exporting.
        return decomposition, None, (f"{MMPBSA_DECOMP_PER_FRAME_FILE_NAME} has no TOTAL "
                                     f"column, so the contributions are tabulated but "
                                     f"not charted.")
    # Only the residues that matter: a long tail of near-zero contributions
    # would leave the significant ones unreadable.
    methods = list(decomposition["Method"].drop_duplicates())
    label_column = "Residue"
    if len(methods) > 1:
        label_column = "Residue (method)"
        decomposition = decomposition.assign(
            **{label_column: decomposition["Residue"] + " (" + decomposition["Method"] + ")"}
        )
    strongest = decomposition.reindex(
        decomposition["TOTAL"].abs().sort_values(ascending=False).index
    ).head(MMPBSA_DECOMPOSITION_RESIDUES_SHOWN).sort_values("TOTAL")
    colours, legend = mmpbsa_residue_colours(strongest["Residue"])
    figure = make_bar_figure(strongest, label_column, "TOTAL", "TOTAL SD",
                             ylabel="ΔG contribution (kcal/mol)",
                             title=(f"Strongest {len(strongest)} residue contributions: "
                                    + ", ".join(methods)),
                             colors=colours, legend=legend)
    return decomposition, figure, ""

def _load_binding_energy_series(results_directory_path: str, working_directory_path: str,
                                structure_file_name: str, input_traj_file_name: str,
                                mmpbsa_input_file_name: str) -> tuple[Any, str]:
    """Binding energy against simulation time, if the per-frame file was written.

    gmx_MMPBSA numbers its frames 1..N over the ones it selected, so the x axis
    is recovered from the trajectory using the startframe and interval the run
    asked for. Falls back to the frame number when the trajectory cannot be
    read, since a plot against frame number still beats no plot.
    """
    per_frame_path = os.path.join(results_directory_path, MMPBSA_PER_FRAME_FILE_NAME)
    if not os.path.exists(per_frame_path):
        return None, ""

    per_frame = parse_mmpbsa_per_frame(per_frame_path)
    method_frames = [method_frame.reset_index(drop=True) for _, method_frame
                     in per_frame.groupby("Method", sort=False)]
    methods = [str(method_frame["Method"].iloc[0]) for method_frame in method_frames]
    reference_frames = method_frames[0]["Frame #"].to_numpy()
    if any(not np.array_equal(method_frame["Frame #"].to_numpy(), reference_frames)
           for method_frame in method_frames[1:]):
        raise ValueError("The MM-PBSA methods contain different frame selections.")

    note = ""
    times_ns: list[float] = []
    input_file_path = os.path.join(working_directory_path, mmpbsa_input_file_name)
    try:
        start_frame, interval = read_mmpbsa_frame_selection(input_file_path)
        times_ns = get_trajectory_frame_times_ns(
            os.path.join(working_directory_path, structure_file_name),
            os.path.join(working_directory_path, input_traj_file_name),
            start_frame, interval, len(reference_frames))
    except Exception as exc:
        note = (f"Binding energy is plotted against frame number: the times could "
                f"not be read from {input_traj_file_name} ({exc}).")

    if len(times_ns) == len(reference_frames):
        frame = pd.DataFrame({"Time (ns)": times_ns})
        x_column = "Time (ns)"
    else:
        if not note:
            note = (f"Binding energy is plotted against frame number: the trajectory "
                    f"holds fewer frames than the run used.")
        frame = pd.DataFrame({"Frame": reference_frames})
        x_column = "Frame"

    y_columns: list[str] = []
    for method, method_frame in zip(methods, method_frames):
        column = ("ΔG binding (kcal/mol)" if len(methods) == 1
                  else f"ΔG binding ({method}) (kcal/mol)")
        frame[column] = method_frame["TOTAL"].to_numpy()
        y_columns.append(column)

    figure = make_line_figure(
        frame, x_column, y_columns=y_columns, ylabel="ΔG binding (kcal/mol)",
        title="Binding energy over the trajectory: " + ", ".join(methods),
        mean_line=len(methods) == 1)
    return figure, note

def on_export_df(working_directory_path: str, df: pd.DataFrame, file_name: str) -> tuple[list[str], str]:
    """Write an analysis table to CSV inside the job directory."""
    if df is None:
        # One export button per analysis now, so exporting before running the
        # matching analysis is an easy mistake to make.
        return get_files_in_working_directory(working_directory_path), \
            "<span style='color:red;'>Run the analysis before exporting its results.</span>"

    try:
        with reserve_working_directory_maintenance(working_directory_path):
            atomic_write_dataframe_csv(
                os.path.join(working_directory_path, file_name), df)
        status = f"File exported: {file_name}"
    except Exception as exc:
        status = "Error exporting file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"  
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

guard_working_directory_reads(globals(), (
    "on_view_protein_structure", "on_view_trajectory",
    "on_analyze_rmsd", "on_analyze_min_distance", "on_analyze_com_distance",
    "on_analyze_rmsf", "on_analyze_sasa", "on_analyze_gyrate", "on_run_pca",
    "on_analyze_free_energy_landscape", "on_load_mmpbsa_results",
))
secure_module_callbacks(globals())


def protein_ligand_complex_md_simulation_tab_content() -> None:
    """Build the Protein-Ligand Complex MD Simulation tab and wire up its callbacks."""
    with gr.Tab(label="Protein-Ligand Complex MD Simulation") as protein_ligand_complex_md_simulation_tab:
        with gr.Row():
            with gr.Column(scale=1):
                working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(), value="md", allow_custom_value=True)
                working_directory_path_state = gr.State()
                open_working_directory_button = gr.Button(value="Create/Open Working Directory")
                working_directory_file_list_state = gr.State()
                working_directory_file_dataframe = gr.Dataframe(label="Files in Working Directory", headers=["File", "Type", "Modified"], max_height=360, interactive=False)
                selected_file_state = gr.State()
                selected_structure_file_state = gr.State()
                selected_text_file_state = gr.State()
                with gr.Row():
                    delete_file_button = gr.Button(value="Delete Selected File", interactive=False)
                    clean_working_directory_button = gr.Button(value="Clean Working Directory", interactive=False)
                with gr.Accordion(label="Structure Viewer", open=False) as structure_viewer_accordion:
                    view_structure_button = gr.Button(value="View Structure", interactive=False)
                    structure_viewer_status_markdown = gr.Markdown()
                    structure_viewer_html = gr.HTML()
                with gr.Accordion(label="Trajectory Viewer", open=False):
                    trajectory_viewer_structure_file_dropdown = gr.Dropdown(label="Structure File", choices=[], value=None)
                    trajectory_viewer_trajectory_file_dropdown = gr.Dropdown(label="Trajectory File", choices=[], value=None)
                    trajectory_viewer_selection_dropdown = gr.Dropdown(label="Selection", choices=list(TRAJECTORY_VIEWER_SELECTIONS), value="Protein + Ligand + Ions")
                    trajectory_viewer_max_frames_slider = gr.Slider(label="Max Frames", minimum=10, maximum=1000, value=200, step=10)
                    view_trajectory_button = gr.Button(value="View Trajectory")
                    trajectory_viewer_status_markdown = gr.Markdown()
                    trajectory_viewer_html = gr.HTML()
                with gr.Accordion(label="Text File Viewer", open=False) as text_file_viewer_accordion:
                    view_text_file_button = gr.Button(value="View Text File", interactive=False)
                    text_file_viewer_textarea = gr.TextArea(label="Text File Viewer", lines=20, elem_id="textfile_viewer", interactive=False)
                    save_text_file_button = gr.Button(value="Save Text File", interactive=False)
            with gr.Column(scale=2):
                with gr.Row():
                    status_markdown = gr.Markdown()
                with gr.Accordion(label="Settings", open=False):
                    with gr.Row():
                        mpi_rank_slider = gr.Slider(label="MPI Ranks", minimum=1, maximum=get_default_cpu_count(), value=1, step=1)
                        omp_threads_slider = gr.Slider(label="OpenMP Threads", minimum=1, maximum=128, value=1, step=1)
                        max_warns_slider = gr.Slider(label="Max Warnings (expert/dangerous override)", minimum=0, maximum=10, value=0, step=1)
                        # CPU is the portable default; explicit GPU task flags
                        # cannot fall back when no accelerator is available.
                        use_gpu = gr.Checkbox(label="Use GPU", value=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion(label="Upload Protein Structure", open=True):
                            with gr.Row():
                                protein_structure_file_name_textbox = gr.Textbox(label="Protein File Name", value="protein.pdb")
                                protein_structure_file = gr.File(label="Upload Protein Structure File", file_types=['.pdb'], interactive=False)
                    with gr.Column(scale=1):
                        with gr.Accordion(label="Upload Ligand Structure", open=True):
                            with gr.Row():
                                ligand_structure_file_name_textbox = gr.Textbox(label="Ligand File Name", value="ligand.pdb")
                                ligand_residue_name_textbox = gr.Textbox(label="Ligand Residue Name (in the uploaded file)", value=LIGAND_RESNAME)
                                ligand_structure_file = gr.File(label="Upload Ligand Structure File", file_types=['.pdb'], interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion(label="Generate Protein Topology", open=False):
                            with gr.Row():
                                with gr.Column():
                                    protein_topology_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    protein_topology_output_file_name_textbox = gr.Textbox(label="Output File Name", value="protein.gro")
                                    protein_topology_output_topology_file_name_textbox = gr.Textbox(label="Output Topology File Name", value="topology.top")
                                with gr.Column():
                                    protein_force_field_dropdown = gr.Dropdown(label="Force Field", choices=["AMBER94", "AMBER96", "AMBER99", "AMBER99SB", "AMBER99SB-ILDN", "AMBER03", ("AMBERGS", "amberGS"), "AMBER14SB", "AMBER19SB",
                                                                                                    "CHARMM27", "CHARMM36", "GROMOS43A1", "GROMOS43A2", "GROMOS45A3", "GROMOS53A5", "GROMOS53A6", "GROMOS54A7", ("OPLS-AA", "OPLSAA")], value="AMBER99SB-ILDN", allow_custom_value=True)
                                    water_model_dropdown = gr.Dropdown(label="Water Model", choices=_WATER_MODEL_CHOICES, value="TIP3P")
                                    n_terminus_dropdown = gr.Dropdown(label="N-Terminus", choices=N_TERMINUS_CHOICES, value=DEFAULT_TERMINUS_CHOICE, allow_custom_value=True)
                                    c_terminus_dropdown = gr.Dropdown(label="C-Terminus", choices=C_TERMINUS_CHOICES, value=DEFAULT_TERMINUS_CHOICE, allow_custom_value=True)
                                    generate_protein_topology_button = gr.Button(value="Generate Topology")
                    with gr.Column(scale=1):
                        with gr.Accordion(label="Generate Ligand Topology", open=False):
                            with gr.Row():
                                with gr.Column():
                                    ligand_topology_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    ligand_output_file_name_textbox = gr.Textbox(label="Output File Name", value="ligand")
                                    ligand_charge_slider = gr.Slider(label="Ligand Charge", minimum=-3, maximum=3, value=0, step=1)
                                    ligand_charge_model_dropdown = gr.Dropdown(label="Charge Model", choices=[("AM1-BCC charges", "bcc"), ("AM1 Mulliken", "gas")], value="bcc")
                                    ligand_force_field_dropdown = gr.Dropdown(label="Ligand Force Field", choices=[("GAFF1", "gaff"), ("GAFF2", "gaff2")], value="gaff")
                                    generate_ligand_topology_button = gr.Button(value="Generate Topology")
                                    gr.Markdown("GAFF is an AMBER-family force field. Pairing it with a CHARMM protein force field is inconsistent; CGenFF is the CHARMM-compatible choice for ligands.")
                with gr.Accordion(label="Merge Structures and Topolopies", open=False):
                    with gr.Row():
                        with gr.Column():
                            merge_structures_protein_input_file_name_dropdown = gr.Dropdown(label="Protein Structure File Name", choices=[], value=None)
                            merge_structures_ligand_input_file_name_dropdown = gr.Dropdown(label="Ligand Structure File Name", choices=[], value=None)
                            merge_structures_output_file_name_textbox = gr.Textbox(label="Output File Name", value="complex.gro")
                            merge_structures_button = gr.Button("Merge structure")
                        with gr.Column():
                            merge_topologies_protein_input_file_name_dropdown = gr.Dropdown(label="Protein Topology File Name", choices=[], value=None)
                            merge_topologies_ligand_input_file_name_dropdown = gr.Dropdown(label="Ligand Topology File Name", choices=[], value=None)
                            merge_topologies_output_file_name_textbox = gr.Textbox(label="Output File Name", value="complex_topology.top")
                            merge_topologies_button = gr.Button("Merge topology")
                with gr.Accordion(label="Generate Simulation Box", open=False):
                    with gr.Row():
                        with gr.Column():
                            box_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            box_output_file_name_textbox = gr.Textbox(label="Output File Name", value="boxed_complex.gro")
                        with gr.Column():
                            box_type_dropdown = gr.Dropdown(label="Box Type", choices=["cubic", "triclinic", "dodecahedron", "octahedron"], value="dodecahedron")
                            distance_slider = gr.Slider(label="Distance to Box Edge (nm)", minimum=1.0, maximum=5.0, value=1.0, step=0.1)
                            generate_box_button = gr.Button(value="Generate Simulation Box")
                with gr.Accordion(label="Solvation", open=False):
                    with gr.Row():
                        with gr.Column():
                            solvation_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            solvation_output_file_name_textbox = gr.Textbox(label="Output File Name", value="solvated_complex.gro")
                            solvation_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                            solvation_output_topology_file_name_textbox = gr.Textbox(label="Output Topology File Name", value="solvated_topology.top")
                        with gr.Column():
                            solvent_configuration_dropdown = gr.Dropdown(label="Solvent Coordinates (set by water model)", choices=["spc216.gro", "tip4p.gro", "tip5p.gro"], value="spc216.gro", interactive=False)
                            solvate_button = gr.Button(value="Solvate Protein")
                with gr.Accordion(label="Add Ions", open=False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for ion addition***")
                            with gr.Row():
                                generate_ions_parameter_file_name_textbox = gr.Textbox(label="Parameter File Name", value="ions.mdp")
                                generate_ions_parameter_file_button = gr.Button(value="Generate Parameter File")
                    with gr.Row():
                        with gr.Column():        
                            with gr.Row():
                                gr.Markdown("***Generate run input file for ion addition***")
                            with gr.Row():
                                with gr.Column():
                                    generate_ions_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    generate_ions_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Column():
                                    generate_ions_parameter_file_dropdown = gr.Dropdown(label="Parameter File Name", choices=[], value=None)
                                    generate_ions_run_input_file_name_textbox = gr.Textbox(label="Run Input File Name", value="ions.tpr")
                                    generate_ions_run_input_file_button = gr.Button(value="Generate Run Input File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Ion addition***")
                            with gr.Row():
                                with gr.Column():
                                    generate_ions_run_input_file_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                                    generate_ions_output_file_name_textbox = gr.Textbox(label="Output File Name", value="ions_complex.gro")
                                    generate_ions_output_topology_file_name_textbox = gr.Textbox(label="Output Topology File Name", value="ions_topology.top")
                                with gr.Column():
                                    cation_name_textbox = gr.Textbox(label="Cation Name", value="NA")
                                    anion_name_textbox = gr.Textbox(label="Anion Name", value="CL")
                                    add_ion_method_radio = gr.Radio(label="Add Ions By", choices=["Concentration", "Number"], value="Concentration")
                                    concentration_slider = gr.Slider(label="Ion Concentration (mM)", minimum=0, maximum=1000, value=150, step=10)
                                    cation_charge_slider = gr.Slider(label="Cation Charge", minimum=1, maximum=3, value=1, step=1, visible=True)
                                    anion_charge_slider = gr.Slider(label="Anion Charge", minimum=-3, maximum=-1, value=-1, step=1, visible=True)
                                    number_of_cations_slider = gr.Slider(label="Number of Cations", minimum=0, maximum=100, value=5, step=1, visible=False)
                                    number_of_anions_slider = gr.Slider(label="Number of Anions", minimum=0, maximum=100, value=5, step=1, visible=False)
                                    netralize_checkbox = gr.Checkbox(label="Neutralize System", value=True)
                                    add_ions_button = gr.Button(value="Add Ions")
                with gr.Accordion(label="Energy Minimization", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for energy minimization***")
                            with gr.Row():
                                energy_minimization_parameter_file_name_textbox = gr.Textbox(label="Parameter File Name", value="em.mdp")
                                energy_minimization_parameter_file_button = gr.Button(value="Generate Parameter File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate run input file for energy minimization***")
                            with gr.Row():        
                                with gr.Column():
                                    energy_minimization_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    energy_minimization_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Column():
                                    energy_minimization_parameter_file_dropdown = gr.Dropdown(label="Parameter File Name", choices=[], value=None)
                                    energy_minimization_run_input_file_name_textbox = gr.Textbox(label="Run Input File Name", value="em.tpr")
                                    energy_minimization_run_input_file_button = gr.Button(value="Generate Run Input File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Run energy minimization***")
                            with gr.Row():
                                with gr.Column():
                                    energy_minimization_run_input_file_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                                with gr.Column():
                                    run_energy_minimization_button = gr.Button(value="Run Energy Minimization")
                with gr.Accordion(label="NVT Equilibration", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for NVT equilibration***")
                            with gr.Row():
                                with gr.Column():
                                    nvt_time_scale_slider = gr.Slider(label="NVT Equilibration Time (ps)", minimum=100, maximum=5000, value=500, step=100)
                                    nvt_time_step_slider = gr.Slider(label="Time Step (ps; no HMR)", minimum=0.001, maximum=0.002, value=0.002, step=0.001)
                                    nvt_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                with gr.Column():
                                    nvt_equilibration_parameter_file_name_textbox = gr.Textbox(label="Parameter File Name", value="nvt.mdp")
                                    nvt_equilibration_parameter_file_button = gr.Button(value="Generate Parameter File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate run input file for NVT equilibration***")
                            with gr.Row():        
                                with gr.Column():
                                    nvt_equilibration_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    nvt_equilibration_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Column():
                                    nvt_equilibration_parameter_file_dropdown = gr.Dropdown(label="Parameter File Name", choices=[], value=None)
                                    nvt_equilibration_run_input_file_name_textbox = gr.Textbox(label="Run Input File Name", value="nvt.tpr")
                                    nvt_equilibration_run_input_file_button = gr.Button(value="Generate Run Input File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Run NVT equilibration***")
                            with gr.Row():
                                with gr.Column():
                                    nvt_equilibration_run_input_file_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                                with gr.Column():
                                    nvt_process_state = gr.State(ProcessStateDict())
                                    run_nvt_equilibration_button = gr.Button(value="Run NVT Equilibration")
                                    nvt_equilibration_timer = gr.Timer(1.0, active=False)
                with gr.Accordion(label="NPT Equilibration", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for NPT equilibration***")
                            with gr.Row():
                                with gr.Column():
                                    npt_time_scale_slider = gr.Slider(label="NPT Equilibration Time (ps)", minimum=100, maximum=5000, value=1000, step=100)
                                    npt_time_step_slider = gr.Slider(label="Time Step (ps; no HMR)", minimum=0.001, maximum=0.002, value=0.002, step=0.001)
                                    npt_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                    npt_pressure_slider = gr.Slider(label="Pressure (bar)", minimum=0.1, maximum=10, value=1, step=0.1)
                                with gr.Column():
                                    npt_equilibration_parameter_file_name_textbox = gr.Textbox(label="Parameter File Name", value="npt.mdp")
                                    npt_equilibration_parameter_file_button = gr.Button(value="Generate Parameter File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate run input file for NPT equilibration***")
                            with gr.Row():        
                                with gr.Column():
                                    npt_equilibration_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    npt_equilibration_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Column():
                                    npt_equilibration_parameter_file_dropdown = gr.Dropdown(label="Parameter File Name", choices=[], value=None)
                                    npt_equilibration_run_input_file_name_textbox = gr.Textbox(label="Run Input File Name", value="npt.tpr")
                                    npt_equilibration_run_input_file_button = gr.Button(value="Generate Run Input File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Run NPT equilibration***")
                            with gr.Row():
                                with gr.Column():
                                    npt_equilibration_run_input_file_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                                with gr.Column():
                                    npt_process_state = gr.State(ProcessStateDict())
                                    run_npt_equilibration_button = gr.Button(value="Run NPT Equilibration")
                                    npt_equilibration_timer = gr.Timer(1.0, active=False)
                with gr.Accordion(label="Production MD", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for production MD simulation***")
                            with gr.Row():
                                with gr.Column():
                                    with gr.Row():
                                        with gr.Column():
                                            prod_md_time_scale_slider = gr.Slider(label="Production MD Time (ns)", minimum=0.001, maximum=1000, value=500, step=1)
                                            prod_md_time_step_slider = gr.Slider(label="Time Step (ps; no HMR)", minimum=0.001, maximum=0.002, value=0.002, step=0.001)
                                            prod_md_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                            prod_md_pressure_slider = gr.Slider(label="Pressure (bar)", minimum=0.1, maximum=10, value=1, step=0.1)
                                    with gr.Row():
                                        prod_md_nnpot_active_checkbox = gr.Checkbox(label="Use Machine Learning Potential (NNPot)", value=False)
                                        prod_md_nnpot_model_dropdown = gr.Dropdown(label="Model", choices=list(SUPPORTED_NNPOT_MODELS), value="ani2x")
                                        prod_md_nnpot_input_group_textbox = gr.Textbox(label="NNPot Input Group", value="Protein")
                                with gr.Column():
                                    prod_md_mdp_type_radio = gr.Radio(label="Initial or continuation", choices=["Initial", "Continuation"], value="Initial")
                                    prod_md_random_seed_textbox = gr.Textbox(label="Random seed", value="0", visible=False)
                                    prod_md_parameter_file_name_textbox = gr.Textbox(label="Parameter File Name", value="md_initial.mdp")
                                    prod_md_parameter_file_button = gr.Button(value="Generate Parameter File")                                    
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                gr.Markdown("***Generate run input file for production MD simulation***")
                            with gr.Row():        
                                with gr.Column():
                                    prod_md_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                                    prod_md_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Column():
                                    prod_md_parameter_file_dropdown = gr.Dropdown(label="Parameter File Name", choices=[], value=None)
                                    prod_md_run_input_file_name_textbox = gr.Textbox(label="Run Input File Name", value="md.tpr")
                                    prod_md_run_input_file_button = gr.Button(value="Generate Run Input File")
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Run production MD simulation***")
                            with gr.Row():
                                with gr.Column():
                                    prod_md_run_input_file_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                                with gr.Column():
                                    gr.Markdown("*Run from beginning*")
                                    prod_md_initial_process_state = gr.State(ProcessStateDict())
                                    run_prod_md_button = gr.Button(value="Run production MD simulation")
                                    prod_md_timer = gr.Timer(1.0, active=False)
                                with gr.Column():
                                    gr.Markdown("*Resume an interrupted run (does not extend the TPR duration)*")
                                    # Both actions write the same -deffnm prefix,
                                    # so they are two controls for one server job.
                                    prod_md_continuation_process_state = prod_md_initial_process_state
                                    checkpoint_file_dropdown = gr.Dropdown(label="Checkpoint File Name", choices=[], value=None)
                                    continue_prod_md_button = gr.Button(value="Resume interrupted production MD")
                with gr.Accordion(label="Fix MD Trajectory", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):    
                            fix_traj_run_input_file_name_dropdown = gr.Dropdown(label="Run Input File Name", choices=[], value=None)
                        with gr.Column(scale=3):    
                            with gr.Row():
                                gr.Markdown("***Make molecules whole***")
                            with gr.Row():   
                                make_mol_whole_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                                make_mol_whole_output_traj_file_name_textbox = gr.Textbox(label="Output Trajectory File Name", value="md_whole.xtc")
                                make_mol_whole_button = gr.Button("Run")
                            with gr.Row():
                                gr.Markdown("***Center protein in the box***")
                            with gr.Row():    
                                center_protein_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                                center_protein_output_traj_file_name_textbox = gr.Textbox(label="Output Trajectory File Name", value="md_center.xtc")  
                                center_protein_button = gr.Button("Run")
                            with gr.Row():
                                gr.Markdown("***Fit to protein backbone***")
                            with gr.Row():     
                                fit_backbone_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)   
                                fit_backbone_output_traj_file_name_textbox = gr.Textbox(label="Output Trajectory File Name", value="md_fit.xtc")
                                fit_backbone_button = gr.Button("Run")
                with gr.Accordion(label="MD Trajectory Analysis", open=False):
                    # Shared inputs on the left, one collapsible block per analysis on
                    # the right: the same shape as Fix MD Trajectory above, which is
                    # what lets this hold many analyses without squeezing them.
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_structure_file_name_dropdown = gr.Dropdown(label="Structure File Name", choices=[], value=None)
                            analysis_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                            analysis_run_input_file_name_dropdown = gr.Dropdown(label="Run Input File Name (.tpr)", choices=[], value=None)
                        with gr.Column(scale=3):
                            with gr.Accordion(label="RMSD (protein and ligand)", open=True):
                                with gr.Row():
                                    rmsd_analyze_button = gr.Button("Run", variant="primary")
                                rmsd_df_state = gr.State()
                                rmsd_plot = gr.Plot()
                                with gr.Row():
                                    rmsd_file_name_texbox = gr.Textbox(label="RMSD File Name", value="RMSD.csv")
                                    rmsd_export_button = gr.Button("Export RMSD (.csv)")
                            with gr.Accordion(label="Minimum distance", open=False):
                                with gr.Row():
                                    min_dist_analyze_button = gr.Button("Run", variant="primary")
                                min_dist_df_state = gr.State()
                                min_dist_plot = gr.Plot()
                                with gr.Row():
                                    min_dist_file_name_texbox = gr.Textbox(label="Minimum Distance File Name", value="Minimum_distance.csv")
                                    min_dist_export_button = gr.Button("Export minimum distance (.csv)")
                            with gr.Accordion(label="Center of mass distance", open=False):
                                with gr.Row():
                                    com_dist_analyze_button = gr.Button("Run", variant="primary")
                                com_dist_df_state = gr.State()
                                com_dist_plot = gr.Plot()
                                with gr.Row():
                                    com_dist_file_name_texbox = gr.Textbox(label="COM Distance File Name", value="COM_distance.csv")
                                    com_dist_export_button = gr.Button("Export COM Distance (.csv)")
                            with gr.Accordion(label="Cα RMSF", open=False):
                                with gr.Row():
                                    ca_rmsf_analyze_button = gr.Button("Run", variant="primary")
                                ca_rmsf_df_state = gr.State()
                                ca_rmsf_plot = gr.Plot()
                                with gr.Row():
                                    ca_rmsf_file_name_texbox = gr.Textbox(label="Cα RMSF File Name", value="C_alpha_RMSF.csv")
                                    ca_rmsf_export_button = gr.Button("Export Cα RMSF (.csv)")
                            with gr.Accordion(label="Solvent Accessible Surface Area", open=False):
                                with gr.Row():
                                    sasa_surface_selection_textbox = gr.Textbox(label="Surface Selection", value=f"group Protein or resname {LIGAND_RESNAME}", info="A bare word is read as an index group whose name can span several words, so combine with the explicit form: group Protein or resname LIG")
                                    sasa_output_selection_textbox = gr.Textbox(label="Output Selection (optional)", value=f"resname {LIGAND_RESNAME}")
                                    sasa_probe_radius_slider = gr.Slider(label="Probe Radius (nm)", minimum=0.05, maximum=0.30, value=0.14, step=0.01)
                                with gr.Row():
                                    sasa_output_file_name_textbox = gr.Textbox(label="Area File Name", value="sasa.xvg")
                                    sasa_residue_output_file_name_textbox = gr.Textbox(label="Per-residue File Name", value="sasa_residue.xvg")
                                    sasa_analyze_button = gr.Button("Run", variant="primary")
                                with gr.Row():
                                    with gr.Column():
                                        sasa_df_state = gr.State()
                                        sasa_plot = gr.Plot()
                                        with gr.Row():
                                            sasa_file_name_texbox = gr.Textbox(label="SASA File Name", value="SASA.csv")
                                            sasa_export_button = gr.Button("Export SASA (.csv)")
                                    with gr.Column():
                                        sasa_residue_df_state = gr.State()
                                        sasa_residue_plot = gr.Plot()
                                        with gr.Row():
                                            sasa_residue_file_name_texbox = gr.Textbox(label="Per-residue File Name", value="SASA_per_residue.csv")
                                            sasa_residue_export_button = gr.Button("Export per-residue (.csv)")
                            with gr.Accordion(label="Radius of Gyration", open=False):
                                with gr.Row():
                                    gyrate_selection_textbox = gr.Textbox(label="Selection", value="group Protein", info="A bare word is read as an index group whose name can span several words, so combine with the explicit form: group Protein or resname LIG")
                                    gyrate_mode_dropdown = gr.Dropdown(label="Weighting", choices=["mass", "charge", "geometry"], value="mass")
                                    gyrate_output_file_name_textbox = gr.Textbox(label="Output File Name", value="gyrate.xvg")
                                    gyrate_analyze_button = gr.Button("Run", variant="primary")
                                gyrate_df_state = gr.State()
                                gyrate_plot = gr.Plot()
                                with gr.Row():
                                    gyrate_file_name_texbox = gr.Textbox(label="Gyration File Name", value="Radius_of_gyration.csv")
                                    gyrate_export_button = gr.Button("Export gyration (.csv)")
                            with gr.Accordion(label="Principal Component Analysis", open=False):
                                with gr.Row():
                                    pca_selection_textbox = gr.Textbox(label="Selection", value="group Backbone", info="A bare word is read as an index group whose name can span several words, so combine with the explicit form: group Protein or resname LIG")
                                    pca_first_eigenvector_slider = gr.Slider(label="First Eigenvector", minimum=1, maximum=10, value=1, step=1)
                                    pca_second_eigenvector_slider = gr.Slider(label="Second Eigenvector", minimum=2, maximum=20, value=2, step=1)
                                    pca_analyze_button = gr.Button("Run", variant="primary")
                                with gr.Row():
                                    pca_index_file_name_textbox = gr.Textbox(label="Index File Name", value="pca_index.ndx")
                                    pca_eigenvector_file_name_textbox = gr.Textbox(label="Eigenvector File Name", value="pca_eigenvec.trr")
                                    pca_eigenvalue_file_name_textbox = gr.Textbox(label="Eigenvalue File Name", value="pca_eigenval.xvg")
                                    pca_projection_file_name_textbox = gr.Textbox(label="Projection File Name", value="pca_2dproj.xvg")
                                with gr.Row():
                                    with gr.Column():
                                        pca_eigenvalue_df_state = gr.State()
                                        pca_eigenvalue_plot = gr.Plot()
                                        with gr.Row():
                                            pca_eigenvalue_file_name_texbox = gr.Textbox(label="Eigenvalue File Name", value="PCA_eigenvalues.csv")
                                            pca_eigenvalue_export_button = gr.Button("Export eigenvalues (.csv)")
                                    with gr.Column():
                                        pca_projection_df_state = gr.State()
                                        pca_projection_plot = gr.Plot()
                                        with gr.Row():
                                            pca_projection_file_name_texbox = gr.Textbox(label="Projection File Name", value="PCA_projection.csv")
                                            pca_projection_export_button = gr.Button("Export projection (.csv)")
                            with gr.Accordion(label="Gibbs Free Energy Landscape", open=False):
                                with gr.Row():
                                    fel_projection_file_name_textbox = gr.Textbox(label="Projection File Name", value="pca_2dproj.xvg")
                                    fel_temperature_slider = gr.Slider(label="Temperature (K)", minimum=100, maximum=500, value=300, step=1)
                                    fel_bin_count_slider = gr.Slider(label="Bins", minimum=20, maximum=200, value=100, step=10)
                                    fel_analyze_button = gr.Button("Run", variant="primary")
                                fel_df_state = gr.State()
                                fel_plot = gr.Plot()
                                with gr.Row():
                                    fel_file_name_texbox = gr.Textbox(label="Landscape File Name", value="Free_energy_landscape.csv")
                                    fel_export_button = gr.Button("Export landscape (.csv)")
                            with gr.Accordion(label="MM-PBSA / MM-GBSA Binding Energy", open=False):
                                mmpbsa_availability_markdown = gr.Markdown(get_gmx_mmpbsa_unavailable_reason() or "")
                                with gr.Row():
                                    mmpbsa_receptor_selection_textbox = gr.Textbox(label="Receptor Selection", value="group Protein", info="A bare word is read as an index group whose name can span several words, so combine with the explicit form: group Protein or resname LIG")
                                    mmpbsa_ligand_selection_textbox = gr.Textbox(label="Ligand Selection", value=f"resname {LIGAND_RESNAME}")
                                    mmpbsa_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                                with gr.Row():
                                    mmpbsa_method_checkboxgroup = gr.CheckboxGroup(label="Method", choices=["MM-GBSA", "MM-PBSA"], value=["MM-GBSA"])
                                    mmpbsa_start_frame_textbox = gr.Textbox(label="Start Frame", value="1")
                                    mmpbsa_end_frame_textbox = gr.Textbox(label="End Frame (0 = last)", value="0")
                                    mmpbsa_interval_slider = gr.Slider(label="Interval (use every Nth frame)", minimum=1, maximum=200, value=100, step=1)
                                with gr.Row():
                                    mmpbsa_temperature_slider = gr.Slider(label="Temperature (K)", minimum=100, maximum=500, value=300, step=1)
                                    mmpbsa_salt_concentration_slider = gr.Slider(label="Salt Concentration (M)", minimum=0.0, maximum=1.0, value=0.15, step=0.01)
                                    mmpbsa_input_file_name_textbox = gr.Textbox(label="Input File Name", value="mmpbsa.in")
                                with gr.Row():
                                    mmpbsa_decomposition_checkbox = gr.Checkbox(label="Per-residue decomposition", value=True)
                                    mmpbsa_decomposition_scheme_dropdown = gr.Dropdown(label="Decomposition Scheme", choices=list(MMPBSA_DECOMPOSITION_SCHEMES), value=2)
                                    mmpbsa_print_residues_textbox = gr.Textbox(label="Residues to Report", value="within 6", info="A gmx_MMPBSA residue selection, e.g. 'within 6' for everything within 6 A of the ligand")
                                    mmpbsa_input_file_button = gr.Button("Generate input file")
                                with gr.Row():
                                    mmpbsa_index_file_name_textbox = gr.Textbox(label="Index File Name", value="mmpbsa_index.ndx")
                                    mmpbsa_processes_slider = gr.Slider(label="MPI Processes", minimum=1, maximum=get_default_cpu_count(), value=1, step=1)
                                    mmpbsa_process_state = gr.State(ProcessStateDict())
                                    run_mmpbsa_button = gr.Button("Start", variant="primary")
                                    mmpbsa_timer = gr.Timer(1.0, active=False)
                                with gr.Row():
                                    mmpbsa_results_file_name_dropdown = gr.Dropdown(label="Results File Name", choices=[], value=None)
                                    mmpbsa_load_button = gr.Button("Load results")
                                # The energy decomposition takes the whole width; the
                                # two per-frame views share the row beneath it.
                                mmpbsa_df_state = gr.State()
                                mmpbsa_plot = gr.Plot()
                                with gr.Row():
                                    mmpbsa_file_name_texbox = gr.Textbox(label="Binding Energy File Name", value="MMPBSA_binding_energy.csv")
                                    mmpbsa_export_button = gr.Button("Export binding energy (.csv)")
                                with gr.Row():
                                    # Left: how the binding energy moves through the
                                    # run. Right: the same numbers as a distribution,
                                    # where a wide or bimodal shape says the mean is
                                    # not the whole story.
                                    mmpbsa_time_series_plot = gr.Plot()
                                    mmpbsa_histogram_plot = gr.Plot()
                                mmpbsa_decomposition_df_state = gr.State()
                                mmpbsa_decomposition_plot = gr.Plot()
                                with gr.Row():
                                    mmpbsa_decomposition_file_name_texbox = gr.Textbox(label="Residue Contribution File Name", value="MMPBSA_residue_contribution.csv")
                                    mmpbsa_decomposition_export_button = gr.Button("Export residue contribution (.csv)")

    # Working directory interactions
    working_directory_open_outputs = [
        working_directory_dropdown, working_directory_path_state,
        working_directory_file_list_state, clean_working_directory_button,
        protein_structure_file, ligand_structure_file,
        selected_file_state, selected_structure_file_state, selected_text_file_state,
        delete_file_button, view_structure_button,
        structure_viewer_status_markdown, structure_viewer_html,
        trajectory_viewer_status_markdown, trajectory_viewer_html,
        view_text_file_button, text_file_viewer_textarea, save_text_file_button,
        status_markdown,
        rmsd_df_state, rmsd_plot,
        min_dist_df_state, min_dist_plot,
        com_dist_df_state, com_dist_plot,
        ca_rmsf_df_state, ca_rmsf_plot,
        sasa_df_state, sasa_plot,
        sasa_residue_df_state, sasa_residue_plot,
        gyrate_df_state, gyrate_plot,
        pca_eigenvalue_df_state, pca_eigenvalue_plot,
        pca_projection_df_state, pca_projection_plot,
        fel_df_state, fel_plot,
        mmpbsa_df_state, mmpbsa_plot,
        mmpbsa_time_series_plot, mmpbsa_histogram_plot,
        mmpbsa_decomposition_df_state, mmpbsa_decomposition_plot,
        run_nvt_equilibration_button, run_npt_equilibration_button,
        run_prod_md_button, continue_prod_md_button, run_mmpbsa_button,
    ]
    working_directory_dropdown.change(on_open_working_directory_and_reset_ui,
                                      working_directory_dropdown,
                                      working_directory_open_outputs)
    open_working_directory_button.click(on_open_working_directory_and_reset_ui,
                                        working_directory_dropdown,
                                        working_directory_open_outputs)
    working_directory_file_list_state.change(on_file_list_change, [working_directory_path_state,
                                                                   protein_structure_file_name_textbox, ligand_structure_file_name_textbox, protein_topology_output_file_name_textbox, ligand_output_file_name_textbox, protein_topology_output_topology_file_name_textbox,
                                                                   merge_structures_output_file_name_textbox, box_output_file_name_textbox, merge_topologies_output_file_name_textbox,
                                                                   solvation_output_file_name_textbox, solvation_output_topology_file_name_textbox,
                                                                   generate_ions_parameter_file_name_textbox, generate_ions_run_input_file_name_textbox, generate_ions_output_file_name_textbox, generate_ions_output_topology_file_name_textbox,
                                                                   energy_minimization_parameter_file_name_textbox, energy_minimization_run_input_file_name_textbox,
                                                                   nvt_equilibration_parameter_file_name_textbox, nvt_equilibration_run_input_file_name_textbox,
                                                                   npt_equilibration_parameter_file_name_textbox, npt_equilibration_run_input_file_name_textbox,
                                                                   prod_md_parameter_file_name_textbox, prod_md_run_input_file_name_textbox,
                                                                   make_mol_whole_output_traj_file_name_textbox, center_protein_output_traj_file_name_textbox, fit_backbone_output_traj_file_name_textbox],
                                             [working_directory_file_dataframe, protein_topology_input_file_name_dropdown, ligand_topology_input_file_name_dropdown,
                                              merge_structures_protein_input_file_name_dropdown, merge_structures_ligand_input_file_name_dropdown, merge_topologies_protein_input_file_name_dropdown, merge_topologies_ligand_input_file_name_dropdown,
                                              box_input_file_name_dropdown, solvation_input_file_name_dropdown, solvation_input_topology_file_name_dropdown,
                                              generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_dropdown,
                                              energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_dropdown,
                                              nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_dropdown,
                                              npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_dropdown,
                                              prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_dropdown, checkpoint_file_dropdown,
                                              fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, center_protein_input_traj_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown,
                                              analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown,
                                              trajectory_viewer_structure_file_dropdown, trajectory_viewer_trajectory_file_dropdown,
                                              analysis_run_input_file_name_dropdown, mmpbsa_input_topology_file_name_dropdown,
                                              mmpbsa_results_file_name_dropdown])
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, [view_structure_button, structure_viewer_accordion])
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, [view_text_file_button, text_file_viewer_accordion])
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    clean_working_directory_button.click(on_clean_working_directory, working_directory_path_state, working_directory_file_list_state)
    view_structure_button.click(on_view_protein_structure, [working_directory_path_state, selected_structure_file_state], [structure_viewer_html, structure_viewer_status_markdown])
    view_trajectory_button.click(on_view_trajectory, [working_directory_path_state, trajectory_viewer_structure_file_dropdown, trajectory_viewer_trajectory_file_dropdown, trajectory_viewer_selection_dropdown, trajectory_viewer_max_frames_slider], [trajectory_viewer_html, trajectory_viewer_status_markdown])
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, selected_text_file_state, text_file_viewer_textarea], working_directory_file_list_state)

    # Protein and ligand structure file upload interaction
    protein_structure_file.upload(on_upload_protein_structure_file, [working_directory_path_state, protein_structure_file_name_textbox, protein_structure_file], [working_directory_file_list_state, status_markdown])
    ligand_structure_file.upload(on_upload_ligand_structure_file, [working_directory_path_state, ligand_structure_file_name_textbox, ligand_residue_name_textbox, ligand_structure_file], [working_directory_file_list_state, status_markdown])
    
    # Generate protein and ligand topology interaction
    generate_protein_topology_button.click(on_generate_protein_topology, [working_directory_path_state, protein_topology_input_file_name_dropdown, protein_topology_output_file_name_textbox, protein_topology_output_topology_file_name_textbox, protein_force_field_dropdown, water_model_dropdown, n_terminus_dropdown, c_terminus_dropdown], [working_directory_file_list_state, status_markdown])
    generate_ligand_topology_button.click(on_generate_ligand_topology, [working_directory_path_state, ligand_topology_input_file_name_dropdown, ligand_output_file_name_textbox, ligand_charge_slider, ligand_charge_model_dropdown, ligand_force_field_dropdown, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])

    # Merge structure and topology interaction
    merge_structures_button.click(on_merge_structures, [working_directory_path_state, merge_structures_protein_input_file_name_dropdown, merge_structures_ligand_input_file_name_dropdown, merge_structures_output_file_name_textbox, merge_topologies_ligand_input_file_name_dropdown], [working_directory_file_list_state, status_markdown])
    merge_topologies_button.click(on_merge_topologies, [working_directory_path_state, merge_topologies_protein_input_file_name_dropdown, merge_topologies_ligand_input_file_name_dropdown, merge_topologies_output_file_name_textbox, protein_force_field_dropdown, merge_structures_ligand_input_file_name_dropdown], [working_directory_file_list_state, status_markdown])

    # Generate simulation box interaction
    generate_box_button.click(on_generate_simulation_box, [working_directory_path_state, box_input_file_name_dropdown, box_output_file_name_textbox, box_type_dropdown, distance_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    protein_force_field_dropdown.change(
        on_force_field_change,
        [protein_force_field_dropdown, distance_slider, water_model_dropdown],
        [distance_slider, water_model_dropdown, solvent_configuration_dropdown],
    )

    # Solvation interaction
    water_model_dropdown.change(on_water_model_change, water_model_dropdown, solvent_configuration_dropdown)
    solvate_button.click(on_solvate_protein, [working_directory_path_state, solvation_input_file_name_dropdown, solvation_output_file_name_textbox, solvation_input_topology_file_name_dropdown, solvation_output_topology_file_name_textbox, solvent_configuration_dropdown, water_model_dropdown], [working_directory_file_list_state, status_markdown])

    # Generate ions interaction
    generate_ions_parameter_file_button.click(on_generate_ions_mdp_file, [working_directory_path_state, generate_ions_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    generate_ions_run_input_file_button.click(on_generate_ions_tpr_file, [working_directory_path_state, generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_name_textbox, max_warns_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    add_ion_method_radio.change(on_add_ions_method_change, add_ion_method_radio, [concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider])
    add_ions_button.click(on_add_ions, [working_directory_path_state, generate_ions_run_input_file_dropdown, generate_ions_output_file_name_textbox, generate_ions_input_topology_file_name_dropdown, generate_ions_output_topology_file_name_textbox, cation_name_textbox, anion_name_textbox, add_ion_method_radio, concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider, netralize_checkbox], [working_directory_file_list_state, status_markdown])
    
    # Energy minimization interaction
    energy_minimization_parameter_file_button.click(on_generate_energy_minimization_mdp_file, [working_directory_path_state, energy_minimization_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    energy_minimization_run_input_file_button.click(on_generate_energy_minimization_tpr_file, [working_directory_path_state, energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_name_textbox, max_warns_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    run_energy_minimization_button.click(on_run_energy_minimization, [working_directory_path_state, energy_minimization_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu], [working_directory_file_list_state, status_markdown])

    # NVT equilibration interaction
    nvt_equilibration_parameter_file_button.click(on_generate_nvt_equilibration_mdp_file, [working_directory_path_state, nvt_time_scale_slider, nvt_time_step_slider, nvt_temperature_slider, nvt_equilibration_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    nvt_equilibration_run_input_file_button.click(on_generate_nvt_equilibration_tpr_file, [working_directory_path_state, nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_name_textbox, max_warns_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    nvt_run_event = run_nvt_equilibration_button.click(on_run_nvt_equilibration, [working_directory_path_state, nvt_equilibration_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, nvt_process_state], [working_directory_file_list_state, status_markdown, nvt_process_state, run_nvt_equilibration_button])
    nvt_run_event.then(_process_timer_update, nvt_process_state,
                       nvt_equilibration_timer, queue=False)
    nvt_equilibration_timer.tick(_sync_process_state_with_timer,
        [working_directory_path_state, nvt_process_state],
        [working_directory_file_list_state, status_markdown,
         run_nvt_equilibration_button, nvt_equilibration_timer])

    # NPT equilibration interaction
    npt_equilibration_parameter_file_button.click(on_generate_npt_equilibration_mdp_file, [working_directory_path_state, npt_time_scale_slider, npt_time_step_slider, npt_temperature_slider, npt_pressure_slider, npt_equilibration_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    npt_equilibration_run_input_file_button.click(on_generate_npt_equilibration_tpr_file, [working_directory_path_state, npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_name_textbox, max_warns_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    npt_run_event = run_npt_equilibration_button.click(on_run_npt_equilibration, [working_directory_path_state, npt_equilibration_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, npt_process_state], [working_directory_file_list_state, status_markdown, npt_process_state, run_npt_equilibration_button])
    npt_run_event.then(_process_timer_update, npt_process_state,
                       npt_equilibration_timer, queue=False)
    npt_equilibration_timer.tick(_sync_process_state_with_timer,
        [working_directory_path_state, npt_process_state],
        [working_directory_file_list_state, status_markdown,
         run_npt_equilibration_button, npt_equilibration_timer])

    # Production MD interaction
    prod_md_mdp_type_radio.change(on_change_mdp_type, prod_md_mdp_type_radio, [prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox])
    prod_md_nnpot_active_checkbox.change(on_toggle_nnpot, [prod_md_nnpot_active_checkbox, prod_md_nnpot_model_dropdown], status_markdown)
    prod_md_nnpot_model_dropdown.change(on_toggle_nnpot, [prod_md_nnpot_active_checkbox, prod_md_nnpot_model_dropdown], status_markdown)
    prod_md_parameter_file_button.click(on_generate_prod_md_mdp_file, [working_directory_path_state, prod_md_time_scale_slider, prod_md_time_step_slider, prod_md_temperature_slider, prod_md_pressure_slider, prod_md_mdp_type_radio, prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox, prod_md_nnpot_active_checkbox, prod_md_nnpot_model_dropdown, prod_md_nnpot_input_group_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    prod_md_run_input_file_button.click(on_generate_prod_md_tpr_file, [working_directory_path_state, prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_name_textbox, max_warns_slider, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    prod_run_event = run_prod_md_button.click(on_run_prod_md, [working_directory_path_state, prod_md_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, prod_md_nnpot_active_checkbox, use_gpu, prod_md_initial_process_state], [working_directory_file_list_state, status_markdown, prod_md_initial_process_state, run_prod_md_button])
    prod_run_event.then(_process_timer_update, prod_md_initial_process_state,
                        prod_md_timer, queue=False)
    prod_continue_event = continue_prod_md_button.click(on_continue_prod_md, [working_directory_path_state, prod_md_run_input_file_dropdown, checkpoint_file_dropdown, mpi_rank_slider, omp_threads_slider, prod_md_nnpot_active_checkbox, use_gpu, prod_md_continuation_process_state], [working_directory_file_list_state, status_markdown, prod_md_continuation_process_state, continue_prod_md_button])
    prod_continue_event.then(_process_timer_update,
                             prod_md_continuation_process_state,
                             prod_md_timer, queue=False)
    prod_md_timer.tick(_sync_shared_process_state_with_timer,
        [working_directory_path_state, prod_md_continuation_process_state],
        [working_directory_file_list_state, status_markdown,
         run_prod_md_button, continue_prod_md_button, prod_md_timer])

    # Fix trajectory interaction
    make_mol_whole_button.click(on_make_molecule_whole, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, make_mol_whole_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    center_protein_button.click(on_center_protein, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, center_protein_input_traj_file_name_dropdown, center_protein_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    fit_backbone_button.click(on_fit_backbone, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown, fit_backbone_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])

    # Analysis
    rmsd_analyze_button.click(on_analyze_rmsd, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, analysis_run_input_file_name_dropdown], [rmsd_df_state, rmsd_plot, status_markdown])
    min_dist_analyze_button.click(on_analyze_min_distance, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [min_dist_df_state, min_dist_plot, status_markdown])
    com_dist_analyze_button.click(on_analyze_com_distance, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, analysis_run_input_file_name_dropdown], [com_dist_df_state, com_dist_plot, status_markdown])
    ca_rmsf_analyze_button.click(on_analyze_rmsf, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, analysis_run_input_file_name_dropdown], [ca_rmsf_df_state, ca_rmsf_plot, status_markdown])
    sasa_analyze_button.click(on_analyze_sasa, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, sasa_surface_selection_textbox, sasa_output_selection_textbox, sasa_probe_radius_slider, sasa_output_file_name_textbox, sasa_residue_output_file_name_textbox], [working_directory_file_list_state, sasa_df_state, sasa_plot, sasa_residue_df_state, sasa_residue_plot, status_markdown])
    mmpbsa_input_file_button.click(on_generate_mmpbsa_input_file, [working_directory_path_state, mmpbsa_input_file_name_textbox, mmpbsa_start_frame_textbox, mmpbsa_end_frame_textbox, mmpbsa_interval_slider, mmpbsa_salt_concentration_slider, mmpbsa_temperature_slider, mmpbsa_method_checkboxgroup, mmpbsa_decomposition_checkbox, mmpbsa_decomposition_scheme_dropdown, mmpbsa_print_residues_textbox], [working_directory_file_list_state, status_markdown])
    mmpbsa_run_event = run_mmpbsa_button.click(on_run_mmpbsa, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, mmpbsa_input_topology_file_name_dropdown, mmpbsa_input_file_name_textbox, mmpbsa_index_file_name_textbox, mmpbsa_receptor_selection_textbox, mmpbsa_ligand_selection_textbox, mmpbsa_processes_slider, mmpbsa_process_state], [working_directory_file_list_state, status_markdown, mmpbsa_process_state, run_mmpbsa_button])
    mmpbsa_run_event.then(_process_timer_update, mmpbsa_process_state,
                          mmpbsa_timer, queue=False)
    mmpbsa_timer.tick(_sync_process_state_with_timer,
        [working_directory_path_state, mmpbsa_process_state],
        [working_directory_file_list_state, status_markdown,
         run_mmpbsa_button, mmpbsa_timer])
    mmpbsa_load_button.click(on_load_mmpbsa_results, [working_directory_path_state, mmpbsa_results_file_name_dropdown, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, mmpbsa_input_file_name_textbox], [working_directory_file_list_state, mmpbsa_df_state, mmpbsa_plot, mmpbsa_time_series_plot, mmpbsa_histogram_plot, mmpbsa_decomposition_df_state, mmpbsa_decomposition_plot, status_markdown])
    pca_analyze_button.click(on_run_pca, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, pca_selection_textbox, pca_first_eigenvector_slider, pca_second_eigenvector_slider, pca_index_file_name_textbox, pca_eigenvector_file_name_textbox, pca_eigenvalue_file_name_textbox, pca_projection_file_name_textbox], [working_directory_file_list_state, pca_eigenvalue_df_state, pca_eigenvalue_plot, pca_projection_df_state, pca_projection_plot, status_markdown])
    fel_analyze_button.click(on_analyze_free_energy_landscape, [working_directory_path_state, fel_projection_file_name_textbox, fel_temperature_slider, fel_bin_count_slider], [fel_df_state, fel_plot, status_markdown])
    gyrate_analyze_button.click(on_analyze_gyrate, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, gyrate_selection_textbox, gyrate_mode_dropdown, gyrate_output_file_name_textbox], [working_directory_file_list_state, gyrate_df_state, gyrate_plot, status_markdown])
    rmsd_export_button.click(on_export_df, [working_directory_path_state, rmsd_df_state, rmsd_file_name_texbox], [working_directory_file_list_state, status_markdown])
    ca_rmsf_export_button.click(on_export_df, [working_directory_path_state, ca_rmsf_df_state, ca_rmsf_file_name_texbox], [working_directory_file_list_state, status_markdown])
    sasa_export_button.click(on_export_df, [working_directory_path_state, sasa_df_state, sasa_file_name_texbox], [working_directory_file_list_state, status_markdown])
    sasa_residue_export_button.click(on_export_df, [working_directory_path_state, sasa_residue_df_state, sasa_residue_file_name_texbox], [working_directory_file_list_state, status_markdown])
    mmpbsa_export_button.click(on_export_df, [working_directory_path_state, mmpbsa_df_state, mmpbsa_file_name_texbox], [working_directory_file_list_state, status_markdown])
    mmpbsa_decomposition_export_button.click(on_export_df, [working_directory_path_state, mmpbsa_decomposition_df_state, mmpbsa_decomposition_file_name_texbox], [working_directory_file_list_state, status_markdown])
    pca_eigenvalue_export_button.click(on_export_df, [working_directory_path_state, pca_eigenvalue_df_state, pca_eigenvalue_file_name_texbox], [working_directory_file_list_state, status_markdown])
    pca_projection_export_button.click(on_export_df, [working_directory_path_state, pca_projection_df_state, pca_projection_file_name_texbox], [working_directory_file_list_state, status_markdown])
    fel_export_button.click(on_export_df, [working_directory_path_state, fel_df_state, fel_file_name_texbox], [working_directory_file_list_state, status_markdown])
    gyrate_export_button.click(on_export_df, [working_directory_path_state, gyrate_df_state, gyrate_file_name_texbox], [working_directory_file_list_state, status_markdown])
    min_dist_export_button.click(on_export_df, [working_directory_path_state, min_dist_df_state, min_dist_file_name_texbox], [working_directory_file_list_state, status_markdown])
    com_dist_export_button.click(on_export_df, [working_directory_path_state, com_dist_df_state, com_dist_file_name_texbox], [working_directory_file_list_state, status_markdown])
    
    return protein_ligand_complex_md_simulation_tab
