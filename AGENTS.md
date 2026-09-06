# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

GROMACS WebUI is a Gradio + FastAPI web application for running molecular dynamics simulations using GROMACS. It provides two main workflows:
1. **Protein MD Simulation** - Single protein simulations
2. **Protein-Ligand Complex MD Simulation** - Protein-ligand binding simulations

The UI guides users through the complete MD workflow: topology generation → solvation → ion addition → energy minimization → equilibration → production MD → trajectory analysis.

## Architecture

### Tech Stack
- **Frontend**: Gradio 4.x (Python-based web UI framework)
- **Backend**: FastAPI with Uvicorn
- **MD Engine**: GROMACS CLI tools (`gmx` commands)
- **Analysis**: MDAnalysis, ParmEd, NGLView
- **Execution**: subprocess-based GROMACS job management with async process watching via daemon threads

### Core Files

- **`webui.py`** - Entry point. Creates FastAPI app, mounts Gradio app, starts Uvicorn server on auto-detected port (starting at 7860).
- **`protein_md_simulation.py`** (1462 lines) - Protein-only simulation workflow. Contains 40+ event handlers for UI interactions.
- **`protein_ligand_complex_md_simulation.py`** (1658 lines) - Protein-ligand simulation workflow. Mirrors protein_md_simulation with added ligand topology generation and merging steps.
- **`utils.py`** - Helper functions: MDP file content generators, protein/ligand manipulation (ParmEd), structure/topology merging.

### Data Flow

1. User uploads protein structure (PDB) via Gradio file input
2. User specifies parameters (force field, water model, temperatures, etc.)
3. Event handlers call GROMACS CLI tools via `subprocess.Popen()`
4. GROMACS output (structures, topologies, trajectories) stored in `./data/<working_directory>/`
5. File list updated after each step (triggers `on_file_list_change()`)
6. Results viewed via NGL Viewer (for structures) or matplotlib (for analysis)
7. MD trajectories analyzed using MDAnalysis, results exported as CSV

### Process Management

- **Long-running jobs** (mdrun equilibration/production) use `subprocess.Popen()` with daemon threads
- Each run (NVT, NPT, Prod MD) has a `process_state` dict shared between main thread and daemon `watch_process()` thread
- `process_state` contains: `{"proc": <Popen object>, "running": <bool>, "lock": <threading.Lock>}`
- **Lock usage** (CRITICAL): All access to `process_state` dict must acquire the lock to prevent race conditions between main thread and daemon watcher thread
- UI timer (`gr.Timer(1.0)`) polls `sync_button_state()` every second to update Run/Stop button based on process state

## Important Patterns & Fixes

### Race Condition Prevention (Threading Lock)
Process state dict is shared between main event handler thread and daemon watcher thread. **Always wrap state access with the lock**:
```python
with process_state["lock"]:
    proc = process_state["proc"]
    process_state["proc"] = None
    process_state["running"] = False
```

### Path Traversal Protection
Working directory paths must be validated to prevent escape from `./data/`:
```python
base = os.path.abspath("./data")
working_directory_path = os.path.abspath(os.path.join("./data/", working_directory))
if not (working_directory_path == base or working_directory_path.startswith(base + os.sep)):
    raise ValueError("Invalid path")
```

### GROMACS Group Selection (genion)
`gmx genion` requires interactive selection of solvent group. Group numbering is topology-dependent (not always group 13). **Use dynamic detection**:
1. Run genion with temp output files using group "0" (System - always exists)
2. Parse stderr to find SOL group number from group listing
3. Run real genion with detected SOL group number
4. Delete temp files

This ensures correctness across different topologies/force fields.

### File Sorting Optimization
In `on_file_list_change()`, sort file listing after building complete list, not inside the loop:
```python
for f in files:
    file_info.append([f, file_type, modified_time])
file_info.sort(...)  # Once after loop, not inside
```

## Running the Application

```bash
# Activate environment
conda activate ./gromacs-env

# Start server
python webui.py
```

Server auto-detects available port (default 7860, increments if busy) and prints URL. Access via browser.

## Directory Structure

```
./data/                  # Working directories for simulations (created at runtime)
./static/                # Generated visualization files (PDB, HTML), cleaned on startup
./gromacs-env/           # Conda environment (conda install artifacts)
protein_md_simulation.py  # Protein workflow
protein_ligand_complex_md_simulation.py  # Protein-ligand workflow
utils.py                 # Shared utilities
webui.py                 # Entry point
```

## Common Development Tasks

**Adding a new GROMACS step:**
1. Create event handler function `on_run_xxx()` in appropriate workflow file
2. If it's a long-running job (mdrun), use subprocess + daemon thread pattern with process_state + lock
3. Update UI layout in `*_tab_content()` function to add buttons/inputs
4. Wire up callbacks in the event binding section at end of file

**Modifying parameter UI:**
- Parameters are `gr.Slider`, `gr.Dropdown`, `gr.Radio` components
- Linked to event handlers via `.click()` or `.change()` callbacks
- Updated file dropdowns by adding to `on_file_list_change()` return values

**Fixing simulation workflow:**
- Most fixes involve GROMACS command building (flags, parameters) in event handlers
- Verify genion group detection works with new topologies
- Test with both protein-only and protein-ligand workflows if changes affect both

## Known Issues / Technical Debt

- Static files (`./static/`) are world-readable (created without restrictive umask)
- Daemon threads may accumulate on rapid start/stop (minimal impact, threads terminate when process finishes)
- Generic exception handling could be more specific (returncode checking vs stderr parsing)

## Dependencies

Critical: GROMACS must be installed and in PATH. Other Python packages:
- `gradio>=4.0` - Web UI framework
- `parmed` - Topology manipulation
- `nglview==4.0` - Structure visualization
- `mdanalysis` - Trajectory analysis
- `acpype` - Ligand topology (conda-forge)
