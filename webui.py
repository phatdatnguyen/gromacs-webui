"""Entry point: mounts the Gradio UI on a FastAPI app and serves it."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import socket

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

import utils
from path_security import (
    DATA_ROOT,
    PROJECT_ROOT,
    STATIC_ROOT,
    cleanup_stale_static_assets,
)
from protein_md_simulation import protein_md_simulation_tab_content
from protein_ligand_complex_md_simulation import protein_ligand_complex_md_simulation_tab_content


LOGGER = logging.getLogger(__name__)
SERVER_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
MAX_PORT_SCAN_ATTEMPTS = 100
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024
STATIC_ASSET_MAX_AGE_SECONDS = 24 * 60 * 60
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 15.0

# Keep all runtime paths anchored to the repository, even when the server is
# started from a different current working directory.
for runtime_root in (DATA_ROOT, STATIC_ROOT):
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkdir's mode is filtered by umask and does not affect an existing path.
    # Both trees can contain uploaded or generated molecular structures.
    runtime_root.chmod(0o700)

@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Perform bounded startup housekeeping and stop child jobs on shutdown."""
    cleanup_stale_static_assets(STATIC_ASSET_MAX_AGE_SECONDS)
    try:
        yield
    finally:
        # Lifespan shutdown runs after request handling has stopped. Invoke the
        # bounded process cleanup directly so interpreter shutdown cannot be
        # held open by an ASGI default-executor worker.
        try:
            utils.stop_all_registered_processes(
                timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
        except Exception:
            # A cleanup failure should be visible without turning an otherwise
            # orderly server exit into a failed ASGI lifespan.
            LOGGER.exception("Failed to stop all registered simulation processes.")


# create a FastAPI app
app = FastAPI(lifespan=application_lifespan)

# create a static directory to store the static files
# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

# function to find an available port
def find_available_port(start_port: int = DEFAULT_PORT,
                        max_attempts: int = MAX_PORT_SCAN_ATTEMPTS) -> int:
    """Return a free loopback port after scanning a bounded port range."""
    if (isinstance(start_port, bool) or not isinstance(start_port, int)
            or not 1 <= start_port <= 65535):
        raise ValueError("start_port must be an integer between 1 and 65535.")
    if (isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
            or max_attempts <= 0):
        raise ValueError("max_attempts must be a positive integer.")

    final_port = min(65535, start_port + max_attempts - 1)
    last_error: OSError | None = None
    for port in range(start_port, final_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((SERVER_HOST, port))
                return port  # Available port found
        except OSError as exc:
            last_error = exc
            continue
    error = RuntimeError(
        f"No available loopback port found from {start_port} through {final_port}."
    )
    if last_error is not None:
        raise error from last_error
    raise error

with gr.Blocks() as blocks:
    with gr.Tabs() as tabs:
        protein_md_simulation_tab_content()
        protein_ligand_complex_md_simulation_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(
    app,
    blocks,
    css_paths=PROJECT_ROOT / "styles.css",
    max_file_size=MAX_UPLOAD_SIZE_BYTES,
    path="/",
)

# serve the app
if __name__ == "__main__":
    available_port = find_available_port()
    uvicorn.run(app, host=SERVER_HOST, port=available_port, access_log=False)
