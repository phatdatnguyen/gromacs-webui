"""Server-side path validation for Gradio callbacks.

Gradio state and dropdown values originate at the client and therefore must not
be treated as trusted merely because the UI normally supplies them.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_ROOT: Path = (PROJECT_ROOT / "data").resolve()
STATIC_ROOT: Path = (PROJECT_ROOT / "static").resolve()
MODEL_ROOT: Path = (PROJECT_ROOT / "models").resolve()

# Only formats deliberately routed to the in-browser text editor.  In
# particular, this keeps a forged callback from reading a multi-gigabyte XTC or
# replacing a TPR with arbitrary text merely because it lives in the job.
EDITABLE_TEXT_EXTENSIONS = frozenset({
    ".top", ".itp", ".mdp", ".log", ".txt", ".dat", ".xvg", ".csv", ".ndx",
})
MAX_EDITABLE_TEXT_BYTES = 16 * 1024 * 1024

# Stored in ``DataFrame.attrs`` while an analysis result lives in ``gr.State``.
# A long analysis can finish after the user has opened another job; provenance
# lets the export callback refuse to write that old result into the new job.
DATAFRAME_WORKING_DIRECTORY_ATTR = "gromacs_webui_working_directory"

_STATIC_ASSET_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_GENERATED_STATIC_PREFIXES = (
    "protein_md_structure_",
    "protein_md_trajectory_",
    "complex_md_structure_",
    "complex_md_trajectory_",
)
MAX_GENERATED_STATIC_ASSET_BYTES = 512 * 1024 * 1024
MAX_GENERATED_STATIC_ASSET_FILES = 300
_STATIC_ASSET_LOCK = threading.Lock()

# File components constrain these roles in the browser, but Gradio callback
# arguments remain client-controlled.  Repeat the contract at the filesystem
# boundary so a mistyped/forged output cannot turn (for example) a TPR into MDP
# text or make trjconv overwrite the trajectory it is still reading.
_STRUCTURE_EXTENSIONS = (".pdb", ".gro")
_TRAJECTORY_EXTENSIONS = (".xtc", ".trr")
_TPR_BUILD_CALLBACKS = (
    "on_generate_ions_tpr_file",
    "on_generate_energy_minimization_tpr_file",
    "on_generate_nvt_equilibration_tpr_file",
    "on_generate_npt_equilibration_tpr_file",
    "on_generate_prod_md_tpr_file",
)
_MDP_BUILD_CALLBACKS = (
    "on_generate_ions_mdp_file",
    "on_generate_energy_minimization_mdp_file",
    "on_generate_nvt_equilibration_mdp_file",
    "on_generate_npt_equilibration_mdp_file",
    "on_generate_prod_md_mdp_file",
)
_MDRUN_CALLBACKS = (
    "on_run_energy_minimization",
    "on_run_nvt_equilibration",
    "on_run_npt_equilibration",
    "on_run_prod_md",
)
_TRAJECTORY_FIX_CALLBACKS = (
    "on_make_molecule_whole", "on_center_protein", "on_fit_backbone",
)

CALLBACK_FILE_EXTENSION_CONTRACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "on_upload_protein_structure_file": {
        "protein_structure_file_name": (".pdb",),
    },
    "on_upload_ligand_structure_file": {
        "ligand_structure_file_name": (".pdb",),
    },
    "on_generate_protein_topology": {
        "input_file_name": _STRUCTURE_EXTENSIONS,
        "output_file_name": (".gro",),
        "output_topology_file_name": (".top",),
    },
    "on_generate_ligand_topology": {
        "ligand_input_file_name": (".pdb",),
        # ACPYPE expands this stem into several related filenames/directories.
        "ligand_output_file_name": ("",),
    },
    "on_merge_structures": {
        "protein_input_file": (".gro",),
        "ligand_input_file": (".gro",),
        "ligand_topology_file": (".itp",),
        "output_file": (".gro",),
    },
    "on_merge_topologies": {
        "protein_input_file": (".top",),
        "ligand_input_file": (".itp",),
        "ligand_structure_file": (".gro",),
        "output_file": (".top",),
    },
    "on_merge_topology": {
        "protein_input_file": (".top",),
        "ligand_input_file": (".itp",),
        "ligand_structure_file": (".gro",),
        "output_file": (".top",),
    },
    "on_generate_simulation_box": {
        "input_file_name": _STRUCTURE_EXTENSIONS,
        "output_file_name": (".gro",),
    },
    "on_solvate_protein": {
        "input_file_name": _STRUCTURE_EXTENSIONS,
        "output_file_name": (".gro",),
        "input_topology_file_name": (".top",),
        "output_topology_file_name": (".top",),
    },
    "on_add_ions": {
        "run_input_file_name": (".tpr",),
        "output_file_name": (".gro",),
        "input_topology_file_name": (".top",),
        "output_topology_file_name": (".top",),
    },
    "on_continue_prod_md": {
        "run_input_file_name": (".tpr",),
        "checkpoint_file_name": (".cpt",),
    },
    "on_view_trajectory": {
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "trajectory_file_name": _TRAJECTORY_EXTENSIONS,
    },
    "on_analyze_rmsd": {
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "run_input_file_name": (".tpr",),
    },
    "on_analyze_rmsf": {
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "run_input_file_name": (".tpr",),
    },
    "on_analyze_min_distance": {
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
    },
    "on_analyze_com_distance": {
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "run_input_file_name": (".tpr",),
    },
    "on_analyze_sasa": {
        "run_input_file_name": (".tpr",),
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "sasa_file_name": (".xvg",),
        "sasa_residue_file_name": (".xvg",),
    },
    "on_analyze_gyrate": {
        "run_input_file_name": (".tpr",),
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "gyrate_file_name": (".xvg",),
    },
    "on_run_pca": {
        "run_input_file_name": (".tpr",),
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "pca_index_file_name": (".ndx",),
        "pca_eigenvector_file_name": (".trr",),
        "pca_eigenvalue_file_name": (".xvg",),
        "pca_projection_file_name": (".xvg",),
    },
    "on_analyze_free_energy_landscape": {
        "projection_file_name": (".xvg",),
    },
    "on_generate_mmpbsa_input_file": {
        "mmpbsa_input_file_name": (".in",),
    },
    "on_run_mmpbsa": {
        "run_input_file_name": (".tpr",),
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "input_topology_file_name": (".top",),
        "mmpbsa_input_file_name": (".in",),
        "mmpbsa_index_file_name": (".ndx",),
    },
    "on_load_mmpbsa_results": {
        "mmpbsa_results_file_name": (".dat",),
        "structure_file_name": _STRUCTURE_EXTENSIONS,
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "mmpbsa_input_file_name": (".in",),
    },
    "on_export_df": {"file_name": (".csv",)},
}
for _callback_name in _TPR_BUILD_CALLBACKS:
    CALLBACK_FILE_EXTENSION_CONTRACTS[_callback_name] = {
        "input_file_name": _STRUCTURE_EXTENSIONS,
        "input_topology_file_name": (".top",),
        "parameter_file_name": (".mdp",),
        "run_input_file_name": (".tpr",),
    }
for _callback_name in _MDP_BUILD_CALLBACKS:
    CALLBACK_FILE_EXTENSION_CONTRACTS[_callback_name] = {
        "parameter_file_name": (".mdp",),
    }
for _callback_name in _MDRUN_CALLBACKS:
    CALLBACK_FILE_EXTENSION_CONTRACTS[_callback_name] = {
        "run_input_file_name": (".tpr",),
    }
for _callback_name in _TRAJECTORY_FIX_CALLBACKS:
    CALLBACK_FILE_EXTENSION_CONTRACTS[_callback_name] = {
        "run_input_file_name": (".tpr",),
        "input_traj_file_name": _TRAJECTORY_EXTENSIONS,
        "output_traj_file_name": _TRAJECTORY_EXTENSIONS,
    }

_DISTINCT_LOCAL_FILE_CALLBACKS = frozenset({
    "on_generate_protein_topology", "on_generate_ligand_topology",
    "on_merge_structures", "on_merge_topologies", "on_merge_topology",
    "on_generate_simulation_box", "on_solvate_protein", "on_add_ions",
    *_TPR_BUILD_CALLBACKS, *_TRAJECTORY_FIX_CALLBACKS,
    "on_analyze_sasa", "on_analyze_gyrate", "on_run_pca",
    "on_run_mmpbsa",
})


def static_asset_basename(prefix: str, working_directory: str | os.PathLike[str]) -> str:
    """Return an invocation-unique basename for generated browser-viewer assets.

    The job digest keeps artifacts recognisable without disclosing the directory
    name.  The random suffix prevents two browser sessions viewing the same job
    from overwriting the PDB/XTC/HTML files underneath one another.
    """
    if not isinstance(prefix, str) or not _STATIC_ASSET_PREFIX_RE.fullmatch(prefix):
        raise ValueError("Invalid static asset prefix.")
    directory = validate_working_directory(working_directory)
    # Run bounded housekeeping for every render, not just once at server start.
    cleanup_stale_static_assets()
    digest = hashlib.sha256(os.fsencode(directory)).hexdigest()[:16]
    return f"{prefix}_{digest}_{secrets.token_hex(8)}"


def cleanup_stale_static_assets(
        max_age_seconds: float = 24 * 60 * 60,
        max_total_bytes: int = MAX_GENERATED_STATIC_ASSET_BYTES,
        max_files: int = MAX_GENERATED_STATIC_ASSET_FILES) -> int:
    """Delete old viewer artifacts and return the number successfully removed.

    Only names generated by this application are eligible.  Symlinks are
    unlinked rather than followed, and unrelated files in ``static`` are kept.
    A one-day grace period avoids breaking viewers in other live browser tabs.
    """
    try:
        max_age_seconds = float(max_age_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Static asset maximum age must be a finite number.") from exc
    if not math.isfinite(max_age_seconds) or max_age_seconds < 0:
        raise ValueError("Static asset maximum age must be finite and non-negative.")
    if (isinstance(max_total_bytes, bool)
            or not isinstance(max_total_bytes, int) or max_total_bytes < 0):
        raise ValueError("Static asset byte limit must be a non-negative integer.")
    if (isinstance(max_files, bool) or not isinstance(max_files, int)
            or max_files < 0):
        raise ValueError("Static asset file limit must be a non-negative integer.")

    cutoff = time.time() - max_age_seconds
    removed = 0
    with _STATIC_ASSET_LOCK:
        try:
            entries = list(STATIC_ROOT.iterdir())
        except OSError:
            # Cleanup is opportunistic and must never prevent server startup.
            return 0

        generated: list[tuple[float, int, Path]] = []
        for entry in entries:
            if not entry.name.startswith(_GENERATED_STATIC_PREFIXES):
                continue
            if entry.suffix.lower() not in {".pdb", ".xtc", ".html"}:
                continue
            try:
                stat = entry.lstat()
                if entry.is_dir() and not entry.is_symlink():
                    continue
                if stat.st_mtime <= cutoff:
                    entry.unlink()
                    removed += 1
                    continue
                generated.append((stat.st_mtime, stat.st_size, entry))
            except FileNotFoundError:
                continue
            except OSError:
                continue

        total_bytes = sum(size for _, size, _ in generated)
        total_files = len(generated)
        for _, size, entry in sorted(generated):
            if total_bytes <= max_total_bytes and total_files <= max_files:
                break
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                continue
            removed += 1
            total_bytes -= size
            total_files -= 1
    return removed


def remove_static_asset_bundle(static_basename: str) -> int:
    """Remove only files belonging to one validated generated viewer basename."""
    if (not isinstance(static_basename, str)
            or Path(static_basename).name != static_basename
            or not static_basename.startswith(_GENERATED_STATIC_PREFIXES)):
        raise ValueError("Invalid generated static asset basename.")
    removed = 0
    with _STATIC_ASSET_LOCK:
        for suffix in (".pdb", ".xtc", ".html", "_view.html"):
            try:
                (STATIC_ROOT / f"{static_basename}{suffix}").unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def tag_dataframe_provenance(value: Any, working_directory: str) -> Any:
    """Return a callback result whose DataFrames are tagged with their source job.

    Callback-owned containers and DataFrames are not mutated.  This matters for
    generator callbacks that may reuse an update dictionary or table in a later
    yield, and for callers that retain their own reference to an analysis table.
    A shallow DataFrame copy is sufficient because only ``attrs`` is changed.
    """
    canonical_directory = validate_working_directory(working_directory)

    def tag(item: Any) -> Any:
        if isinstance(item, pd.DataFrame):
            tagged = item.copy(deep=False)
            tagged.attrs = dict(item.attrs)
            tagged.attrs[DATAFRAME_WORKING_DIRECTORY_ATTR] = canonical_directory
            return tagged
        if isinstance(item, tuple):
            return tuple(tag(nested) for nested in item)
        if isinstance(item, list):
            return [tag(nested) for nested in item]
        if isinstance(item, dict):
            # ``gr.update(...)`` values are ordinary dictionaries.  Preserve
            # component keys and update metadata without modifying the object
            # returned by the callback.
            return {key: tag(nested) for key, nested in item.items()}
        return item

    return tag(value)


def validate_dataframe_provenance(value: Any, working_directory: str) -> None:
    """Reject nested analysis tables that originated in a different job."""
    destination = validate_working_directory(working_directory)

    def validate(item: Any, seen: set[int]) -> None:
        if isinstance(item, pd.DataFrame):
            source = item.attrs.get(DATAFRAME_WORKING_DIRECTORY_ATTR)
            if source is None:
                # Permit callers outside the UI and states created before this
                # safeguard was introduced. Secured callback results are tagged.
                return
            try:
                canonical_source = validate_working_directory(source)
            except ValueError:
                canonical_source = None
            if canonical_source != destination:
                raise ValueError(
                    "This analysis result belongs to a different working directory. "
                    "Run the analysis again in the currently open job before exporting it."
                )
            return

        if not isinstance(item, (tuple, list, dict)) or id(item) in seen:
            return
        seen.add(id(item))
        nested_values = item.values() if isinstance(item, dict) else item
        for nested in nested_values:
            validate(nested, seen)

    validate(value, set())


def validate_working_directory(path: str | os.PathLike[str] | None) -> str:
    """Return an absolute data directory, rejecting escapes (including symlinks)."""
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError("A working directory must be opened first.")

    resolved = Path(path).resolve()
    if resolved != DATA_ROOT and DATA_ROOT not in resolved.parents:
        raise ValueError("Invalid working directory: path must stay inside ./data/.")
    return str(resolved)


def validate_file_name(value: str | None, parameter_name: str = "file name") -> str | None:
    """Accept a single local filename, never an absolute or relative path."""
    if value is None:
        return value
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value in {".", ".."}):
        raise ValueError(f"Invalid {parameter_name}.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"Invalid {parameter_name}: control characters are not allowed.")
    # Several filenames are written into quoted GROMACS #include directives.
    # Reject quote characters rather than letting a name alter the topology.
    if any(character in value for character in ('"', "'", "<", ">")):
        raise ValueError(
            f"Invalid {parameter_name}: quote and angle-bracket characters are not allowed.")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {parameter_name}: directory components are not allowed.")
    return value


def validate_local_file_path(working_directory: str | os.PathLike[str], file_name: str | None,
                             parameter_name: str = "file name") -> str:
    """Reject filenames whose existing symlink target escapes the job directory."""
    if file_name is None:
        raise ValueError(f"Invalid {parameter_name}.")
    validate_file_name(file_name, parameter_name)
    directory = Path(validate_working_directory(working_directory))
    target = (directory / file_name).resolve()
    if target.parent != directory:
        raise ValueError(f"Invalid {parameter_name}: path must stay inside the working directory.")
    return str(target)


def validate_file_extension(file_name: str,
                            expected_extensions: tuple[str, ...],
                            parameter_name: str) -> None:
    """Enforce a callback file role independently of browser-side filtering."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in expected_extensions:
        choices = ", ".join(extension or "an extension-free name"
                            for extension in expected_extensions)
        raise ValueError(
            f"Invalid {parameter_name}: expected {choices}.")


def validate_editable_text_file(working_directory: str | os.PathLike[str],
                                file_name: str | None) -> str:
    """Return an existing, bounded text-editor target inside a job directory."""
    target = validate_local_file_path(
        working_directory, file_name, "text file name")
    suffix = Path(target).suffix.lower()
    if suffix not in EDITABLE_TEXT_EXTENSIONS:
        raise ValueError(
            f"'{file_name}' is not a supported editable text file.")
    try:
        size = os.path.getsize(target)
    except FileNotFoundError:
        raise ValueError(f"Text file '{file_name}' does not exist.") from None
    if not os.path.isfile(target):
        raise ValueError(f"Text file '{file_name}' is not a regular file.")
    if size > MAX_EDITABLE_TEXT_BYTES:
        raise ValueError(
            f"Text file '{file_name}' is too large for the editor "
            f"(maximum {MAX_EDITABLE_TEXT_BYTES // (1024 * 1024)} MiB).")
    return target


def read_editable_text_file(working_directory: str | os.PathLike[str],
                            file_name: str | None) -> str:
    """Read one validated editor file without an unbounded ``read()`` call."""
    target = validate_editable_text_file(working_directory, file_name)
    with open(target, encoding="utf-8", errors="replace") as handle:
        content = handle.read(MAX_EDITABLE_TEXT_BYTES + 1)
    if len(content.encode("utf-8")) > MAX_EDITABLE_TEXT_BYTES:
        raise ValueError(
            f"Text file '{file_name}' is too large for the editor "
            f"(maximum {MAX_EDITABLE_TEXT_BYTES // (1024 * 1024)} MiB).")
    return content


def atomic_replace_editable_text_file(
        working_directory: str | os.PathLike[str], file_name: str | None,
        content: str) -> None:
    """Atomically replace an existing editor file with bounded UTF-8 text."""
    target = validate_editable_text_file(working_directory, file_name)
    if not isinstance(content, str):
        raise ValueError("Text file content must be text.")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_EDITABLE_TEXT_BYTES:
        raise ValueError(
            f"Text content is too large for the editor "
            f"(maximum {MAX_EDITABLE_TEXT_BYTES // (1024 * 1024)} MiB).")

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".text_edit_", suffix=Path(target).suffix,
        dir=os.path.dirname(target))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def secure_working_directory_callback(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Validate path-like callback arguments before any filesystem operation."""
    signature = inspect.signature(callback)
    if "working_directory_path" not in signature.parameters:
        return callback

    def validated(args: Any, kwargs: Any) -> inspect.BoundArguments:
        """Validate the client-supplied paths and return the bound arguments."""
        bound = signature.bind(*args, **kwargs)
        bound.arguments["working_directory_path"] = validate_working_directory(
            bound.arguments["working_directory_path"]
        )
        local_paths: dict[str, str] = {}
        extension_contracts = CALLBACK_FILE_EXTENSION_CONTRACTS.get(
            callback.__name__, {})
        for name, value in bound.arguments.items():
            # Uploaded *_file_path values are Gradio-managed source paths. Every
            # filename used inside the working directory contains "file" but not
            # "path" (including protein_input_file and selected_file_name).
            if "file" in name and "path" not in name and isinstance(value, str):
                local_paths[name] = validate_local_file_path(
                    bound.arguments["working_directory_path"], value, name)
                expected_extensions = extension_contracts.get(name)
                if expected_extensions is not None:
                    validate_file_extension(value, expected_extensions, name)
            validate_dataframe_provenance(
                value, bound.arguments["working_directory_path"])
        if callback.__name__ in _DISTINCT_LOCAL_FILE_CALLBACKS:
            seen: dict[str, str] = {}
            for name, path in local_paths.items():
                prior_name = seen.get(path)
                if prior_name is not None:
                    raise ValueError(
                        f"{name} must be different from {prior_name}; input and "
                        "output files cannot share the same path.")
                seen[path] = name
        return bound

    # A generator callback needs a generator wrapper. functools.wraps sets
    # __wrapped__, but inspect.isgeneratorfunction reads the code flags of the
    # object it is handed and does not follow it, so a plain wrapper would look
    # like an ordinary function to Gradio. Gradio would then treat the returned
    # generator as the output value instead of streaming what it yields, and
    # every streaming analysis would render as an object repr.
    if inspect.isgeneratorfunction(callback):
        @functools.wraps(callback)
        def secured_generator(*args: Any, **kwargs: Any) -> Any:
            bound = validated(args, kwargs)
            directory = bound.arguments["working_directory_path"]
            for result in callback(*bound.args, **bound.kwargs):
                yield tag_dataframe_provenance(result, directory)

        return secured_generator

    if inspect.isasyncgenfunction(callback):
        @functools.wraps(callback)
        async def secured_async_generator(*args: Any, **kwargs: Any) -> Any:
            bound = validated(args, kwargs)
            directory = bound.arguments["working_directory_path"]
            async for result in callback(*bound.args, **bound.kwargs):
                yield tag_dataframe_provenance(result, directory)

        return secured_async_generator

    if inspect.iscoroutinefunction(callback):
        @functools.wraps(callback)
        async def secured_coroutine(*args: Any, **kwargs: Any) -> Any:
            bound = validated(args, kwargs)
            result = await callback(*bound.args, **bound.kwargs)
            return tag_dataframe_provenance(
                result, bound.arguments["working_directory_path"])

        return secured_coroutine

    @functools.wraps(callback)
    def secured(*args: Any, **kwargs: Any) -> Any:
        bound = validated(args, kwargs)
        result = callback(*bound.args, **bound.kwargs)
        return tag_dataframe_provenance(
            result, bound.arguments["working_directory_path"])

    return secured


def secure_module_callbacks(namespace: MutableMapping[str, Any]) -> None:
    """Wrap all already-defined UI callbacks that receive a working directory."""
    for name, callback in list(namespace.items()):
        if name.startswith("on_") and callable(callback):
            namespace[name] = secure_working_directory_callback(callback)
