"""One-shot helper: import torch (and friends) inside the functions that need them."""

path = "utils.py"
src = open(path, encoding="utf-8").read()

replacements = [
    # is_cached_nnpot_model_usable
    ('''def is_cached_nnpot_model_usable(model_name: str, modelfile_path: str) -> bool:
    """Report whether the cached model matches this build, moving it aside if not."""
    extra_files = {"nnpot_model_config": ""}''',
     '''def is_cached_nnpot_model_usable(model_name: str, modelfile_path: str) -> bool:
    """Report whether the cached model matches this build, moving it aside if not."""
    import torch

    extra_files = {"nnpot_model_config": ""}'''),
    # checkExtensions
    ('''def checkExtensions() -> dict[str, str]:
    """Collect loaded Torch extension libraries to embed alongside a saved model."""
    ext_lib = []''',
     '''def checkExtensions() -> dict[str, str]:
    """Collect loaded Torch extension libraries to embed alongside a saved model."""
    import torch

    ext_lib = []'''),
    # trace_aimnet2_model
    ('''def trace_aimnet2_model(model: torch.nn.Module) -> torch.jit.ScriptModule:
    """Trace AIMNet2 with representative inputs, since it cannot be scripted."""
    model.eval()''',
     '''def trace_aimnet2_model(model: torch.nn.Module) -> torch.jit.ScriptModule:
    """Trace AIMNet2 with representative inputs, since it cannot be scripted."""
    import torch

    model.eval()'''),
    # download_nnpot_model: the heavy imports live here
    ('''def download_nnpot_model(model_name: str) -> str:
    """Build or reuse the wrapped neural-network potential and return its file path."""
    os.makedirs("./models", exist_ok=True)''',
     '''def download_nnpot_model(model_name: str) -> str:
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

    os.makedirs("./models", exist_ok=True)'''),
]

for old, new in replacements:
    assert old in src, f"not found:\\n{old[:80]}"
    src = src.replace(old, new, 1)

open(path, "w", encoding="utf-8", newline="\n").write(src)
print("utils.py: torch/e3nn/nnpot_models now imported on demand")
