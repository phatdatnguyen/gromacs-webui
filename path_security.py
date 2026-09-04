"""Server-side path validation for Gradio callbacks.

Gradio state and dropdown values originate at the client and therefore must not
be treated as trusted merely because the UI normally supplies them.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any


DATA_ROOT: Path = (Path(__file__).resolve().parent / "data").resolve()


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
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"Invalid {parameter_name}.")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {parameter_name}: directory components are not allowed.")
    return value


def validate_local_file_path(working_directory: str | os.PathLike[str], file_name: str | None,
                             parameter_name: str = "file name") -> str:
    """Reject filenames whose existing symlink target escapes the job directory."""
    validate_file_name(file_name, parameter_name)
    directory = Path(validate_working_directory(working_directory))
    target = (directory / file_name).resolve()
    if target.parent != directory:
        raise ValueError(f"Invalid {parameter_name}: path must stay inside the working directory.")
    return str(target)


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
        for name, value in bound.arguments.items():
            # Uploaded *_file_path values are Gradio-managed source paths. Every
            # filename used inside the working directory contains "file" but not
            # "path" (including protein_input_file and selected_file_name).
            if "file" in name and "path" not in name and isinstance(value, str):
                validate_local_file_path(bound.arguments["working_directory_path"], value, name)
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
            yield from callback(*bound.args, **bound.kwargs)

        return secured_generator

    @functools.wraps(callback)
    def secured(*args: Any, **kwargs: Any) -> Any:
        bound = validated(args, kwargs)
        return callback(*bound.args, **bound.kwargs)

    return secured


def secure_module_callbacks(namespace: MutableMapping[str, Any]) -> None:
    """Wrap all already-defined UI callbacks that receive a working directory."""
    for name, callback in list(namespace.items()):
        if name.startswith("on_") and callable(callback):
            namespace[name] = secure_working_directory_callback(callback)
