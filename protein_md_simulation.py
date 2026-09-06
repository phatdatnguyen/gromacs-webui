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
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import gradio as gr
import nglview
import MDAnalysis as mda
from MDAnalysis.analysis import rms
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
        # TIP3P is the form's carried default, not evidence of an intentional
        # choice for a newly selected family. Move to that family's published
        # recommendation while preserving any other supported user selection.
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
    # A textbox value can name an existing job-local directory.  Moving such a
    # directory into the backup slot and recursively deleting it after publish
    # would permanently destroy unrelated data (notably custom *.ff trees).
    # Preflight every destination before moving the first file so the operation
    # stays all-or-nothing.
    for staged_path, destination in staged_files:
        if not os.path.lexists(destination):
            continue
        staged_is_directory = os.path.isdir(staged_path)
        destination_is_directory = os.path.isdir(destination)
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


def get_working_directories() -> list[str]:
    """Names of the job directories that already exist under ./data, sorted by name."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted((entry.name for entry in DATA_ROOT.iterdir() if entry.is_dir()), key=str.lower)

def get_files_in_working_directory(working_directory_path: str | None) -> list[str]:
    """Visible files in a job directory, hiding backups and tool scratch files.

    Sorted by name: os.listdir() order is arbitrary, and every file dropdown in
    the UI is filtered straight out of this list. The MM-PBSA scratch files are
    hidden here too: a job directory is browsable from either tab, so the two
    listings have to agree about what is worth showing."""
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
        return None, None, None, None, None

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
        return None, None, None, None, None

    return gr.update(choices=get_working_directories(), value=working_directory), working_directory_path, files, gr.update(interactive=True), gr.update(interactive=True)

def on_file_list_change(working_directory_path: str, protein_structure_file_name: str,
                        topology_output_file_name: str, box_output_file_name: str,
                        topology_output_topology_file_name: str, solvation_output_file_name: str,
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
    topology_files = [f for f in files if f.lower().endswith('.top')]
    parameter_files = [f for f in files if f.lower().endswith('.mdp')]
    run_input_files = [f for f in files if f.lower().endswith('.tpr')]
    checkpoint_files = [f for f in files if f.lower().endswith('.cpt')]
    # Both GROMACS and MDAnalysis accept compressed XTC and full-precision TRR.
    trajectory_files = [f for f in files if f.lower().endswith(('.xtc', '.trr'))]
    viewer_trajectory_files = trajectory_files

    # Update topology input file name dropdown
    if protein_structure_file_name in structure_files:
        topology_input_file_name_value = protein_structure_file_name
    else:
        topology_input_file_name_value = structure_files[0] if structure_files else None

    # Update box input file name dropdown
    if topology_output_file_name in structure_files:
        box_input_file_name_value = topology_output_file_name
    else:
        box_input_file_name_value = structure_files[0] if structure_files else None

    # Update solvation input file dropdown
    if box_output_file_name in structure_files:
        solvation_input_file_name_value = box_output_file_name
    else:
        solvation_input_file_name_value = structure_files[0] if structure_files else None

    # Update solvation input topology file dropdown
    if topology_output_topology_file_name in topology_files:
        solvation_input_topology_file_name_value = topology_output_topology_file_name
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

    # Update nvt equilibration input file dropdown
    energy_minimization_output_structure = (os.path.splitext(energy_minimization_run_input_file_name)[0] + ".gro"
                                            if energy_minimization_run_input_file_name else "")
    if energy_minimization_run_input_file_name in run_input_files and energy_minimization_output_structure in structure_files:
        nvt_equilibration_input_file_name_value = energy_minimization_output_structure
    else:
        nvt_equilibration_input_file_name_value = structure_files[0] if structure_files else None

    # Update energy minimization run input file dropdown
    if energy_minimization_run_input_file_name in run_input_files:
        energy_minimization_run_input_file_name_value = energy_minimization_run_input_file_name
    else:
        energy_minimization_run_input_file_name_value = run_input_files[0] if run_input_files else None

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

    return file_df, \
        gr.update(choices=structure_files, value=topology_input_file_name_value), \
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
        gr.update(choices=run_input_files, value=fix_traj_run_input_file_name_value)

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
        # Analysis data and plots must not remain exportable/displayed for a
        # different working directory.
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
        # A prior job may have left these controls displaying Stop. The process
        # timer will independently detach/reattach state after the directory swap.
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
        static_basename = static_asset_basename("protein_md_structure", working_directory_path)
        structure_path = STATIC_ROOT / f"{static_basename}.pdb"
        viewer_path = STATIC_ROOT / f"{static_basename}.html"
        # Representations follow whatever species the file actually contains, so
        # ions like CU2P are drawn instead of being silently skipped.
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
        static_basename = static_asset_basename("protein_md_trajectory", working_directory_path)
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

def on_generate_protein_topology(working_directory_path: str, input_file_name: str, output_file_name: str,
                                 output_topology_file_name: str, force_field: str, water_model: str,
                                 n_terminus: str, c_terminus: str) -> tuple[list[str], str]:
    """Run pdb2gmx, optionally choosing explicit N- and C-terminus patches."""
    try:
        _validate_force_field_water_model(force_field, water_model)
        # The bundled amberGS directory is case-sensitive; the other displayed
        # force-field names map to their lowercase directory identifiers.
        force_field_id = "amberGS" if force_field.lower() == "ambergs" else force_field.lower()
        # Isolate every pdb2gmx byproduct. The input and any job-local custom
        # force-field directories are made available inside the stage, preserving
        # the prior lookup behavior while keeping failed GRO/TOP/ITP output away
        # from the last known-good files.
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

            # Plain output names keep generated #include directives portable
            # after the complete artifact set is moved into the job root.
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
                # The AMBER ports, for example, patch termini through renamed terminal
                # residues and offer no menu, so run without -ter instead of failing.
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
            # dict.fromkeys avoids listing an explicitly named .itp twice if a
            # future UI permits one as a principal output.
            _publish_staged_files([
                (staged_path,
                 os.path.join(working_directory_path, os.path.basename(staged_path)))
                for staged_path in dict.fromkeys(staged_outputs)
            ])
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
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
            temperature=temperature, force_field=force_field)
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
        running = process_state["running"]
    if running:
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
        # Switching jobs detaches this browser control; it does not terminate
        # the old simulation, whose registry entry and watcher remain active.
        clear_process_state_if_current(process_state, proc)
        return (gr.update(), gr.update(),
                gr.update(value="Start", variant="primary"))

    refresh_process_state(process_state)
    running, message, color, job_directory = consume_process_completion(process_state)
    button = (gr.update(value="Stop", variant="stop") if running
              else gr.update(value="Start", variant="primary"))
    if message is None:
        return gr.update(), gr.update(), button

    # A user can switch working directories while a simulation continues.  Its
    # completion status remains useful, but its files must not replace the list
    # belonging to the directory currently open in the UI.
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
    # If registration failed after Popen, the registry still contains the
    # reserved sentinel rather than ``proc``. This second identity-safe release
    # handles that case and releases its CPU/GPU admission too.
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
            temperature=temperature, pressure=pressure, force_field=force_field)
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
            status = "Error downloading NNPot model!\n" + str(exc)
            return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"

    try:
        file_content = get_default_prod_md_mdp_file_content(time_scale_ps=time_scale*1000, time_step_ps=time_step, temperature=temperature, pressure=pressure, mdp_type=mdp_type, random_seed=random_seed, nnpot_active=nnpot_active, nnpot_modelfile_path=nnpot_modelfile_path, nnpot_input_group=nnpot_input_group, nnpot_model_name=nnpot_model_name, force_field=force_field)
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
                    # With domain decomposition, keep just the generally safe
                    # non-bonded offload instead of constructing an mdrun command
                    # that GROMACS rejects at start-up.
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
        # Initial and continuation runs intentionally use the same key because
        # both write ``-deffnm <base_name>``.  They must never append to the same
        # trajectory/checkpoint at the same time.
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

def on_analyze_rmsd(working_directory_path: str, structure_file_name: str,
                    input_traj_file_name: str,
                    run_input_file_name: str | None = None) -> tuple[Any, ...]:
    """Whole-protein RMSD after fitting its backbone to the first frame."""
    universe = None
    try:
        if run_input_file_name:
            times_ns, rmsd_values = gromacs_backbone_fitted_rmsd(
                working_directory_path, run_input_file_name,
                input_traj_file_name, ["Protein"],
                group_resolver=get_gmx_group_input,
                command_runner=run_checked_command)
            frame = pd.DataFrame({"Time (ns)": times_ns,
                                  "Protein RMSD (Å)": rmsd_values[:, 0]})
            figure = make_line_figure(
                frame, "Time (ns)", ylabel="RMSD (Å)", title="RMSD vs Time")
            status = "RMSD calculated successfully with TPR-aware PBC correction."
            return frame, figure, "<span style='color:green;'>" + status + "</span>"

        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        protein_rmsd = rms.RMSD(
            universe,
            select="protein and backbone",
            groupselections=["protein"],
            ref_frame=0
        ).run()

        values = np.asarray(protein_rmsd.results.rmsd, dtype=float)
        if (values.ndim != 2 or values.shape[1] < 4 or values.shape[0] == 0
                or not np.all(np.isfinite(values[:, [1, 3]]))
                or np.any(np.diff(values[:, 1]) < 0)
                or np.any(values[:, 3] < 0)):
            raise ValueError("MDAnalysis returned invalid RMSD or trajectory time values.")
        frame = pd.DataFrame({"Time (ns)": values[:, 1] / 1000,
                              # Column 2 is the backbone fit RMSD.  The first
                              # group-selection column is the whole protein
                              # measured in that fitted coordinate system.
                              "Protein RMSD (Å)": values[:, 3]})
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
    "on_analyze_rmsd", "on_analyze_rmsf", "on_analyze_sasa",
    "on_analyze_gyrate", "on_run_pca", "on_analyze_free_energy_landscape",
))
secure_module_callbacks(globals())


def protein_md_simulation_tab_content() -> None:
    """Build the Protein MD Simulation tab and wire up its callbacks."""
    with gr.Tab(label="Protein MD Simulation") as protein_md_simulation_tab:
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
                        # CPU is the portable default.  Explicit GPU task flags
                        # make mdrun fail immediately on hosts without a usable
                        # accelerator, while users with one can opt in here.
                        use_gpu = gr.Checkbox(label="Use GPU", value=False)
                with gr.Accordion(label="Upload Protein Structure", open=True):
                    with gr.Row():
                        protein_structure_file_name_textbox = gr.Textbox(label="Protein File Name", value="protein.pdb")
                        protein_structure_file = gr.File(label="Upload Protein Structure File", file_types=['.pdb'], interactive=False)
                with gr.Accordion(label="Generate Protein Topology", open=False):
                    with gr.Row():
                        with gr.Column():
                            topology_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            topology_output_file_name_textbox = gr.Textbox(label="Output File Name", value="protein.gro")
                            topology_output_topology_file_name_textbox = gr.Textbox(label="Output Topology File Name", value="topology.top")
                        with gr.Column():
                            force_field_dropdown = gr.Dropdown(label="Force Field", choices=["AMBER94", "AMBER96", "AMBER99", "AMBER99SB", "AMBER99SB-ILDN", "AMBER03", ("AMBERGS", "amberGS"), "AMBER14SB", "AMBER19SB",
                                                                                            "CHARMM27", "CHARMM36", "GROMOS43A1", "GROMOS43A2", "GROMOS45A3", "GROMOS53A5", "GROMOS53A6", "GROMOS54A7", ("OPLS-AA", "OPLSAA")], value="AMBER99SB-ILDN", allow_custom_value=True)
                            water_model_dropdown = gr.Dropdown(label="Water Model", choices=_WATER_MODEL_CHOICES, value="TIP3P")
                            n_terminus_dropdown = gr.Dropdown(label="N-Terminus", choices=N_TERMINUS_CHOICES, value=DEFAULT_TERMINUS_CHOICE, allow_custom_value=True)
                            c_terminus_dropdown = gr.Dropdown(label="C-Terminus", choices=C_TERMINUS_CHOICES, value=DEFAULT_TERMINUS_CHOICE, allow_custom_value=True)
                            generate_topology_button = gr.Button(value="Generate Topology")
                with gr.Accordion(label="Generate Simulation Box", open=False):
                    with gr.Row():
                        with gr.Column():
                            box_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            box_output_file_name_textbox = gr.Textbox(label="Output File Name", value="boxed_protein.gro")
                        with gr.Column():
                            box_type_dropdown = gr.Dropdown(label="Box Type", choices=["cubic", "triclinic", "dodecahedron", "octahedron"], value="dodecahedron")
                            distance_slider = gr.Slider(label="Distance to Box Edge (nm)", minimum=1.0, maximum=5.0, value=1.0, step=0.1)
                            generate_box_button = gr.Button(value="Generate Simulation Box")
                with gr.Accordion(label="Solvation", open=False):
                    with gr.Row():
                        with gr.Column():
                            solvation_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            solvation_output_file_name_textbox = gr.Textbox(label="Output File Name", value="solvated_protein.gro")
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
                                    generate_ions_output_file_name_textbox = gr.Textbox(label="Output File Name", value="ions_protein.gro")
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
                    with gr.Row():
                        with gr.Column(scale=1):
                            analysis_structure_file_name_dropdown = gr.Dropdown(label="Structure File Name", choices=[], value=None)
                            analysis_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                            analysis_run_input_file_name_dropdown = gr.Dropdown(label="Run Input File Name (.tpr)", choices=[], value=None)
                        with gr.Column(scale=3):
                            with gr.Accordion(label="Protein RMSD", open=True):
                                with gr.Row():
                                    protein_rmsd_analyze_button = gr.Button("Run", variant="primary")
                                protein_rmsd_df_state = gr.State()
                                protein_rmsd_plot = gr.Plot()
                                with gr.Row():
                                    protein_rmsd_file_name_texbox = gr.Textbox(label="Protein RMSD File Name", value="Protein_RMSD.csv")
                                    protein_rmsd_export_button = gr.Button("Export protein RMSD (.csv)")
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
                                    sasa_surface_selection_textbox = gr.Textbox(label="Surface Selection", value="group Protein", info="A bare word is read as an index group whose name can span several words, so combine with the explicit form: group Protein or resname LIG")
                                    sasa_output_selection_textbox = gr.Textbox(label="Output Selection (optional)", value="")
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

    # Working directory interactions
    working_directory_open_outputs = [
        working_directory_dropdown, working_directory_path_state,
        working_directory_file_list_state, clean_working_directory_button,
        protein_structure_file,
        selected_file_state, selected_structure_file_state, selected_text_file_state,
        delete_file_button, view_structure_button,
        structure_viewer_status_markdown, structure_viewer_html,
        trajectory_viewer_status_markdown, trajectory_viewer_html,
        view_text_file_button, text_file_viewer_textarea, save_text_file_button,
        status_markdown,
        protein_rmsd_df_state, protein_rmsd_plot,
        ca_rmsf_df_state, ca_rmsf_plot,
        sasa_df_state, sasa_plot,
        sasa_residue_df_state, sasa_residue_plot,
        gyrate_df_state, gyrate_plot,
        pca_eigenvalue_df_state, pca_eigenvalue_plot,
        pca_projection_df_state, pca_projection_plot,
        fel_df_state, fel_plot,
        run_nvt_equilibration_button, run_npt_equilibration_button,
        run_prod_md_button, continue_prod_md_button,
    ]
    working_directory_dropdown.change(on_open_working_directory_and_reset_ui,
                                      working_directory_dropdown,
                                      working_directory_open_outputs)
    open_working_directory_button.click(on_open_working_directory_and_reset_ui,
                                        working_directory_dropdown,
                                        working_directory_open_outputs)
    working_directory_file_list_state.change(on_file_list_change, [working_directory_path_state,
                                                                   protein_structure_file_name_textbox, topology_output_file_name_textbox, box_output_file_name_textbox, topology_output_topology_file_name_textbox,
                                                                   solvation_output_file_name_textbox, solvation_output_topology_file_name_textbox,
                                                                   generate_ions_parameter_file_name_textbox, generate_ions_run_input_file_name_textbox, generate_ions_output_file_name_textbox, generate_ions_output_topology_file_name_textbox,
                                                                   energy_minimization_parameter_file_name_textbox, energy_minimization_run_input_file_name_textbox,
                                                                   nvt_equilibration_parameter_file_name_textbox, nvt_equilibration_run_input_file_name_textbox,
                                                                   npt_equilibration_parameter_file_name_textbox, npt_equilibration_run_input_file_name_textbox,
                                                                   prod_md_parameter_file_name_textbox, prod_md_run_input_file_name_textbox,
                                                                   make_mol_whole_output_traj_file_name_textbox, center_protein_output_traj_file_name_textbox, fit_backbone_output_traj_file_name_textbox],
                                             [working_directory_file_dataframe, topology_input_file_name_dropdown, box_input_file_name_dropdown,
                                              solvation_input_file_name_dropdown, solvation_input_topology_file_name_dropdown,
                                              generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_dropdown,
                                              energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_dropdown,
                                              nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_dropdown,
                                              npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_dropdown,
                                              prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_dropdown, checkpoint_file_dropdown,
                                              fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, center_protein_input_traj_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown,
                                              analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown,
                                              trajectory_viewer_structure_file_dropdown, trajectory_viewer_trajectory_file_dropdown,
                                              analysis_run_input_file_name_dropdown])
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, [view_structure_button, structure_viewer_accordion])
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, [view_text_file_button, text_file_viewer_accordion])
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    clean_working_directory_button.click(on_clean_working_directory, working_directory_path_state, working_directory_file_list_state)
    view_structure_button.click(on_view_protein_structure, [working_directory_path_state, selected_structure_file_state], [structure_viewer_html, structure_viewer_status_markdown])
    view_trajectory_button.click(on_view_trajectory, [working_directory_path_state, trajectory_viewer_structure_file_dropdown, trajectory_viewer_trajectory_file_dropdown, trajectory_viewer_selection_dropdown, trajectory_viewer_max_frames_slider], [trajectory_viewer_html, trajectory_viewer_status_markdown])
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, selected_text_file_state, text_file_viewer_textarea], working_directory_file_list_state)

    # Protein structure file upload interaction
    protein_structure_file.upload(on_upload_protein_structure_file, [working_directory_path_state, protein_structure_file_name_textbox, protein_structure_file], [working_directory_file_list_state, status_markdown])

    # Generate protein topology interaction
    generate_topology_button.click(on_generate_protein_topology, [working_directory_path_state, topology_input_file_name_dropdown, topology_output_file_name_textbox, topology_output_topology_file_name_textbox, force_field_dropdown, water_model_dropdown, n_terminus_dropdown, c_terminus_dropdown], [working_directory_file_list_state, status_markdown])
    
    # Generate simulation box interaction
    generate_box_button.click(on_generate_simulation_box, [working_directory_path_state, box_input_file_name_dropdown, box_output_file_name_textbox, box_type_dropdown, distance_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    force_field_dropdown.change(
        on_force_field_change,
        [force_field_dropdown, distance_slider, water_model_dropdown],
        [distance_slider, water_model_dropdown, solvent_configuration_dropdown],
    )

    # Solvation interaction
    water_model_dropdown.change(on_water_model_change, water_model_dropdown, solvent_configuration_dropdown)
    solvate_button.click(on_solvate_protein, [working_directory_path_state, solvation_input_file_name_dropdown, solvation_output_file_name_textbox, solvation_input_topology_file_name_dropdown, solvation_output_topology_file_name_textbox, solvent_configuration_dropdown, water_model_dropdown], [working_directory_file_list_state, status_markdown])

    # Generate ions interaction
    generate_ions_parameter_file_button.click(on_generate_ions_mdp_file, [working_directory_path_state, generate_ions_parameter_file_name_textbox, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    generate_ions_run_input_file_button.click(on_generate_ions_tpr_file, [working_directory_path_state, generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_name_textbox, max_warns_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    add_ion_method_radio.change(on_add_ions_method_change, add_ion_method_radio, [concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider])
    add_ions_button.click(on_add_ions, [working_directory_path_state, generate_ions_run_input_file_dropdown, generate_ions_output_file_name_textbox, generate_ions_input_topology_file_name_dropdown, generate_ions_output_topology_file_name_textbox, cation_name_textbox, anion_name_textbox, add_ion_method_radio, concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider, netralize_checkbox], [working_directory_file_list_state, status_markdown])
    
    # Energy minimization interaction
    energy_minimization_parameter_file_button.click(on_generate_energy_minimization_mdp_file, [working_directory_path_state, energy_minimization_parameter_file_name_textbox, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    energy_minimization_run_input_file_button.click(on_generate_energy_minimization_tpr_file, [working_directory_path_state, energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_name_textbox, max_warns_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    run_energy_minimization_button.click(on_run_energy_minimization, [working_directory_path_state, energy_minimization_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu], [working_directory_file_list_state, status_markdown])

    # NVT equilibration interaction
    nvt_equilibration_parameter_file_button.click(on_generate_nvt_equilibration_mdp_file, [working_directory_path_state, nvt_time_scale_slider, nvt_time_step_slider, nvt_temperature_slider, nvt_equilibration_parameter_file_name_textbox, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    nvt_equilibration_run_input_file_button.click(on_generate_nvt_equilibration_tpr_file, [working_directory_path_state, nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_name_textbox, max_warns_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    nvt_run_event = run_nvt_equilibration_button.click(on_run_nvt_equilibration, [working_directory_path_state, nvt_equilibration_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, nvt_process_state], [working_directory_file_list_state, status_markdown, nvt_process_state, run_nvt_equilibration_button])
    nvt_run_event.then(_process_timer_update, nvt_process_state,
                       nvt_equilibration_timer, queue=False)
    nvt_equilibration_timer.tick(_sync_process_state_with_timer,
        [working_directory_path_state, nvt_process_state],
        [working_directory_file_list_state, status_markdown,
         run_nvt_equilibration_button, nvt_equilibration_timer])

    # NPT equilibration interaction
    npt_equilibration_parameter_file_button.click(on_generate_npt_equilibration_mdp_file, [working_directory_path_state, npt_time_scale_slider, npt_time_step_slider, npt_temperature_slider, npt_pressure_slider, npt_equilibration_parameter_file_name_textbox, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    npt_equilibration_run_input_file_button.click(on_generate_npt_equilibration_tpr_file, [working_directory_path_state, npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_name_textbox, max_warns_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
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
    prod_md_parameter_file_button.click(on_generate_prod_md_mdp_file, [working_directory_path_state, prod_md_time_scale_slider, prod_md_time_step_slider, prod_md_temperature_slider, prod_md_pressure_slider, prod_md_mdp_type_radio, prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox, prod_md_nnpot_active_checkbox, prod_md_nnpot_model_dropdown, prod_md_nnpot_input_group_textbox, force_field_dropdown], [working_directory_file_list_state, status_markdown])
    prod_md_run_input_file_button.click(on_generate_prod_md_tpr_file, [working_directory_path_state, prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_name_textbox, max_warns_slider, force_field_dropdown], [working_directory_file_list_state, status_markdown])
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
    protein_rmsd_analyze_button.click(on_analyze_rmsd, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, analysis_run_input_file_name_dropdown], [protein_rmsd_df_state, protein_rmsd_plot, status_markdown])
    ca_rmsf_analyze_button.click(on_analyze_rmsf, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown, analysis_run_input_file_name_dropdown], [ca_rmsf_df_state, ca_rmsf_plot, status_markdown])
    sasa_analyze_button.click(on_analyze_sasa, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, sasa_surface_selection_textbox, sasa_output_selection_textbox, sasa_probe_radius_slider, sasa_output_file_name_textbox, sasa_residue_output_file_name_textbox], [working_directory_file_list_state, sasa_df_state, sasa_plot, sasa_residue_df_state, sasa_residue_plot, status_markdown])
    pca_analyze_button.click(on_run_pca, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, pca_selection_textbox, pca_first_eigenvector_slider, pca_second_eigenvector_slider, pca_index_file_name_textbox, pca_eigenvector_file_name_textbox, pca_eigenvalue_file_name_textbox, pca_projection_file_name_textbox], [working_directory_file_list_state, pca_eigenvalue_df_state, pca_eigenvalue_plot, pca_projection_df_state, pca_projection_plot, status_markdown])
    fel_analyze_button.click(on_analyze_free_energy_landscape, [working_directory_path_state, fel_projection_file_name_textbox, fel_temperature_slider, fel_bin_count_slider], [fel_df_state, fel_plot, status_markdown])
    gyrate_analyze_button.click(on_analyze_gyrate, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, gyrate_selection_textbox, gyrate_mode_dropdown, gyrate_output_file_name_textbox], [working_directory_file_list_state, gyrate_df_state, gyrate_plot, status_markdown])
    protein_rmsd_export_button.click(on_export_df, [working_directory_path_state, protein_rmsd_df_state, protein_rmsd_file_name_texbox], [working_directory_file_list_state, status_markdown])
    ca_rmsf_export_button.click(on_export_df, [working_directory_path_state, ca_rmsf_df_state, ca_rmsf_file_name_texbox], [working_directory_file_list_state, status_markdown])
    sasa_export_button.click(on_export_df, [working_directory_path_state, sasa_df_state, sasa_file_name_texbox], [working_directory_file_list_state, status_markdown])
    sasa_residue_export_button.click(on_export_df, [working_directory_path_state, sasa_residue_df_state, sasa_residue_file_name_texbox], [working_directory_file_list_state, status_markdown])
    pca_eigenvalue_export_button.click(on_export_df, [working_directory_path_state, pca_eigenvalue_df_state, pca_eigenvalue_file_name_texbox], [working_directory_file_list_state, status_markdown])
    pca_projection_export_button.click(on_export_df, [working_directory_path_state, pca_projection_df_state, pca_projection_file_name_texbox], [working_directory_file_list_state, status_markdown])
    fel_export_button.click(on_export_df, [working_directory_path_state, fel_df_state, fel_file_name_texbox], [working_directory_file_list_state, status_markdown])
    gyrate_export_button.click(on_export_df, [working_directory_path_state, gyrate_df_state, gyrate_file_name_texbox], [working_directory_file_list_state, status_markdown])

    return protein_md_simulation_tab
