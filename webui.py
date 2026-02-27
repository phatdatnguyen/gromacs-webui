import os
import glob
import gradio as gr
from protein_md_simulation import protein_md_simulation_tab_content
from protein_ligand_complex_md_simulation import protein_ligand_complex_md_simulation_tab_content
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import socket

# create necessary directories
os.makedirs('./data', exist_ok=True)
os.makedirs('./static', exist_ok=True)

# clean up the directory
for filepath in glob.iglob('./static/*.html'):
    os.remove(filepath)
for filepath in glob.iglob('./static/*.pdb'):
    os.remove(filepath)

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# function to find an available port
def find_available_port(start_port=7860):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port

available_port = find_available_port()

with gr.Blocks() as blocks:
    with gr.Tabs() as tabs:
        protein_md_simulation_tab_content()
        protein_ligand_complex_md_simulation_tab_content()

# mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, blocks, css_paths=Path('./styles.css'), path="/")

# serve the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=available_port, access_log=False)
