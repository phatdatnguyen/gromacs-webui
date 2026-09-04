import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import re
import time
import threading
import psutil
import shutil
import subprocess
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
from path_security import DATA_ROOT, secure_module_callbacks, validate_file_name

def get_working_directories() -> list[str]:
    """Names of the job directories that already exist under ./data, sorted by name."""
    base_path = "./data/"
    os.makedirs(base_path, exist_ok=True)
    return sorted((d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))), key=str.lower)

# gmx_MMPBSA runs in the job directory, because a topology's #include lines only
# resolve beside it, and leaves dozens of these behind. They are working files,
# not results, so they are hidden the same way GROMACS backups are.
MMPBSA_SCRATCH_PREFIX: str = "_GMXMMPBSA_"

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

def on_open_working_directory(working_directory: str | None) -> tuple[Any, ...]:
    """Create or open a job directory under ./data and enable the file actions."""
    if working_directory is None or working_directory.strip() == "":
        gr.Warning("Please specify a working directory.")
        return None, None, None, None, None, None

    try:
        validate_file_name(working_directory, "working directory")
        working_directory_path = str((DATA_ROOT / working_directory).resolve())
    except ValueError as exc:
        gr.Warning(str(exc))
        return None, None, None, None, None, None
    if DATA_ROOT not in Path(working_directory_path).parents:
        gr.Warning("Invalid working directory: path must stay inside ./data/")
        return None, None, None, None, None, None

    os.makedirs(working_directory_path, exist_ok=True)
    files = get_files_in_working_directory(working_directory_path)

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
        if f.endswith('.pdb') or f.endswith('.gro'):
            file_type = "Structure File"
        elif f.endswith('.top') or f.endswith('.itp'):
            file_type = "Topology File"
        elif f.endswith('.mdp'):
            file_type = "Parameter File"
        elif f.endswith('.tpr'):
            file_type = "Run Input File"
        elif f.endswith('.log'):
            file_type = "Log File"
        elif f.endswith('.edr'):
            file_type = "Energy File"
        elif f.endswith('.trr') or f.endswith('.xtc'):
            file_type = "Trajectory File"
        elif f.endswith('.cpt'):
            file_type = "Checkpoint File"
        elif f.endswith('.csv') or f.endswith('.xvg'):
            file_type = "Data File"
        elif f.endswith('.ndx'):
            file_type = "Index File"
        else:
            file_type = "Other File"
        modified_time = time.ctime(os.path.getmtime(file_path))
        file_info.append([f, file_type, modified_time])
    file_info.sort(key=lambda x: x[2].lower(), reverse=True)
    file_df = pd.DataFrame(file_info, columns=["File", "Type", "Modified"])

    # Filter structure and text files
    structure_files = [f for f in files if f.endswith('.pdb') or f.endswith('.gro')]
    topology_files = [f for f in files if f.endswith('.top') or f.endswith('.itp')]
    parameter_files = [f for f in files if f.endswith('.mdp')]
    run_input_files = [f for f in files if f.endswith('.tpr')]
    checkpoint_files = [f for f in files if f.endswith('.cpt')]
    trajectory_files = [f for f in files if f.endswith('.xtc')]
    # NGL and MDAnalysis both read .trr, so the viewer accepts it as well.
    viewer_trajectory_files = [f for f in files if f.endswith('.xtc') or f.endswith('.trr')]
    # gmx_MMPBSA writes its report as .dat, and a job can hold several from
    # different runs, so the results panel picks one rather than being told a name.
    results_files = [f for f in files if f.endswith('.dat')]

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
    if protein_topology_output_file_name in structure_files:
        merge_structure_protein_input_file_name_value = protein_topology_output_file_name
    else:
        merge_structure_protein_input_file_name_value = structure_files[0] if structure_files else None
    
    # Update merge structure ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.gro" in structure_files:
        merge_structure_ligand_input_file_name_value = f"{ligand_output_file_name}_GMX.gro"
    else:
        merge_structure_ligand_input_file_name_value = structure_files[0] if structure_files else None

    # Update merge topology protein input file name dropdown
    if protein_topology_output_topology_file_name in topology_files:
        merge_topology_protein_input_file_name_value = protein_topology_output_topology_file_name
    else:
        merge_topology_protein_input_file_name_value = topology_files[0] if topology_files else None
    
    # Update merge topology ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.itp" in topology_files:
        merge_topology_ligand_input_file_name_value = f"{ligand_output_file_name}_GMX.itp"
    else:
        merge_topology_ligand_input_file_name_value = topology_files[0] if topology_files else None

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
    if energy_minimization_run_input_file_name in run_input_files and f"{energy_minimization_run_input_file_name.split('.')[0]}.gro" in structure_files:
        nvt_equilibration_input_file_name_value = f"{energy_minimization_run_input_file_name.split('.')[0]}.gro"
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
    if nvt_equilibration_run_input_file_name in run_input_files and f"{nvt_equilibration_run_input_file_name.split('.')[0]}.gro" in structure_files:
        npt_equilibration_input_file_name_value = f"{nvt_equilibration_run_input_file_name.split('.')[0]}.gro"
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
    if npt_equilibration_run_input_file_name in run_input_files and f"{npt_equilibration_run_input_file_name.split('.')[0]}.gro" in structure_files:
        prod_md_input_file_name_value = f"{npt_equilibration_run_input_file_name.split('.')[0]}.gro"
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
    if prod_md_run_input_file_name in run_input_files and f"{prod_md_run_input_file_name.split('.')[0]}.cpt" in checkpoint_files:
        prod_md_checkpoint_file_name_value = f"{prod_md_run_input_file_name.split('.')[0]}.cpt"
    else:
        prod_md_checkpoint_file_name_value = checkpoint_files[0] if checkpoint_files else None

    # Update fix trajectory run input file dropdown
    if prod_md_run_input_file_name in run_input_files:
        fix_traj_run_input_file_name_value = prod_md_run_input_file_name
    else:
        fix_traj_run_input_file_name_value = run_input_files[0] if run_input_files else None

    # Update make molecule whole input trajectory file dropdown
    if prod_md_run_input_file_name in run_input_files and f"{prod_md_run_input_file_name.split('.')[0]}.xtc" in trajectory_files:
        make_mol_whole_input_traj_file_name_value = f"{prod_md_run_input_file_name.split('.')[0]}.xtc"
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
    if prod_md_run_input_file_name in run_input_files and f"{prod_md_run_input_file_name.split('.')[0]}.gro" in structure_files:
        analysis_structure_file_name_value = f"{prod_md_run_input_file_name.split('.')[0]}.gro"
    else:
        analysis_structure_file_name_value = structure_files[0] if structure_files else None

    # Update analysis input trajectory file dropdown
    if fit_backbone_output_traj_file_name in trajectory_files:
        analysis_input_traj_file_name_value = fit_backbone_output_traj_file_name
    else:
        analysis_input_traj_file_name_value = trajectory_files[0] if trajectory_files else None

    # Update MM-PBSA results file dropdown, preferring the name gmx_MMPBSA uses.
    if MMPBSA_RESULTS_FILE_NAME in results_files:
        mmpbsa_results_file_name_value = MMPBSA_RESULTS_FILE_NAME
    else:
        mmpbsa_results_file_name_value = results_files[0] if results_files else None

    return file_df, \
        gr.update(choices=structure_files, value=protein_topology_input_file_name_value), \
        gr.update(choices=structure_files, value=ligand_topology_input_file_name_value), \
        gr.update(choices=structure_files, value=merge_structure_protein_input_file_name_value), \
        gr.update(choices=structure_files, value=merge_structure_ligand_input_file_name_value), \
        gr.update(choices=topology_files, value=merge_topology_protein_input_file_name_value), \
        gr.update(choices=topology_files, value=merge_topology_ligand_input_file_name_value), \
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
    if selected_file_name.endswith('.pdb') or selected_file_name.endswith('.gro'):
        return selected_file_name, selected_file_name, None, gr.update(interactive=True)
    elif selected_file_name.endswith('.top') or selected_file_name.endswith('.itp') or selected_file_name.endswith('.mdp') or selected_file_name.endswith('.log'):
        return selected_file_name, None, selected_file_name, gr.update(interactive=True)
    else:
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

def on_delete_file(working_directory_path: str, selected_file_name: str | None) -> list[str]:
    """Delete the selected file and return the refreshed file list."""
    if selected_file_name is None:
        return get_files_in_working_directory(working_directory_path)
    
    file_path = os.path.join(working_directory_path, selected_file_name)
    try:
        os.remove(file_path)
        status = "File deleted successfully."
    except Exception as exc:
        status = "Error deleting file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_clean_working_directory(working_directory_path: str) -> list[str]:
    """Remove GROMACS backup files and Zone.Identifier leftovers."""
    try:
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
    try:
        # Representations follow whatever species the file actually contains, so the
        # ligand is picked up without hardcoding LIG and ions such as CU2P are drawn.
        protein_file_path, species = prepare_structure_viewer_file(
            os.path.join(working_directory_path, protein_file_name),
            "./static/complex_md_structure.pdb",
        )

        # Create the NGL view widget
        view = nglview.show_structure_file(protein_file_path)
        add_species_representations_to_nglview(view, species)

        # Write the widget to HTML
        if os.path.exists('./static/complex_md_structure.html'):
            os.remove('./static/complex_md_structure.html')
        nglview.write_html('./static/complex_md_structure.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/complex_md_structure.html?ts={timestamp}" height="800" width="600" title="NGL View"></iframe>'

        return html, "<span style='color:green;'>" + get_species_legend(species) + "</span>"
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, "<span style='color:red;'>Error loading structure!</span>"

def on_view_trajectory(working_directory_path: str, structure_file_name: str | None,
                       trajectory_file_name: str | None, selection: str,
                       max_frames: int) -> tuple[str | None, str | None]:
    """Reduce the trajectory, then return an iframe that animates it with NGL."""
    if structure_file_name is None or trajectory_file_name is None:
        gr.Warning("Please select both a structure file and a trajectory file.")
        return None, None

    try:
        static_basename = "complex_md_trajectory"
        info = write_trajectory_viewer_files(
            os.path.join(working_directory_path, structure_file_name),
            os.path.join(working_directory_path, trajectory_file_name),
            selection,
            max_frames,
            static_basename,
        )

        viewer_file_path = f"./static/{static_basename}_view.html"
        timestamp = int(time.time())
        with open(viewer_file_path, 'w') as file:
            file.write(get_trajectory_viewer_html(static_basename, timestamp, info["frames"], info["species"]))

        html = f'<iframe src="/static/{static_basename}_view.html?ts={timestamp}" height="800" width="600" title="NGL Trajectory View"></iframe>'
        status = (f"Showing {info['frames']} of {info['total_frames']} frames (every {info['stride']}), "
                  f"{info['n_atoms']} atoms. {get_species_legend(info['species'])}")

        return html, "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, "<span style='color:red;'>Error loading trajectory!</span>"

def on_view_text_file(working_directory_path: str,
                      text_file_name: str) -> tuple[GradioUpdate | None, GradioUpdate | None]:
    """Load a text file into the editor and enable saving it."""
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'r') as file:
            content = file.read()
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
    
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        status = "File saved successfully."
    except Exception as exc:
        status = "Error saving file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_upload_protein_structure_file(working_directory_path: str, protein_structure_file_name: str,
                                     protein_structure_file_path: str) -> tuple[list[str], str]:
    """Copy an uploaded protein structure into the job directory."""
    # Upload and rename the file
    save_file_path = os.path.join(working_directory_path, protein_structure_file_name)
    try:
        if os.path.exists(save_file_path):
            os.remove(save_file_path)

        shutil.copy2(protein_structure_file_path, save_file_path)

        status = "File uploaded successfully."
        return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        status = "Error uploading file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"

def on_upload_ligand_structure_file(working_directory_path: str, ligand_structure_file_name: str,
                                    ligand_residue_name: str,
                                    ligand_structure_file_path: str) -> tuple[list[str], str]:
    """Copy an uploaded ligand structure into the job directory as residue LIG.

    ``ligand_residue_name`` is what the ligand is called in the file being
    uploaded. It only needs changing when the file holds more than the ligand:
    if the name is absent from the file, every atom in it is treated as ligand,
    which is the usual case and covers files whose residue field is empty.
    """
    # Upload and rename the file
    save_file_path = os.path.join(working_directory_path, ligand_structure_file_name)
    try:
        if os.path.exists(save_file_path):
            os.remove(save_file_path)

        shutil.copy2(ligand_structure_file_path, save_file_path)

        # Files in the wild name the molecule UNK, MOL, a component id, or leave
        # the field empty, but the analysis selects the ligand as "resname LIG".
        present = read_pdb_residue_names(save_file_path)
        replaced = rename_pdb_residues(save_file_path, LIGAND_RESNAME, ligand_residue_name)

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
    
def on_generate_protein_topology(working_directory_path: str, input_file_name: str, output_file_name: str,
                                 output_topology_file_name: str, force_field: str, water_model: str,
                                 n_terminus: str, c_terminus: str) -> tuple[list[str], str]:
    """Run pdb2gmx, optionally choosing explicit N- and C-terminus patches."""
    try:
        # Run inside the working directory with plain file names: pdb2gmx writes the
        # -i path verbatim into the topology's "#ifdef POSRES" include, so passing a
        # path here would leave the topology only usable from this app's directory.
        cmd = [
            "gmx", "pdb2gmx",
            "-f", input_file_name,
            "-o", output_file_name,
            "-p", output_topology_file_name,
            "-i", "posre.itp",
            "-ff", force_field.lower(),
            "-water", water_model.lower(),
            "-ignh"
        ]

        select_termini = n_terminus != DEFAULT_TERMINUS_CHOICE or c_terminus != DEFAULT_TERMINUS_CHOICE
        if select_termini:
            cmd.append("-ter")

        print(f"Running command (in {working_directory_path}): {' '.join(cmd)}")

        if select_termini:
            answers, resolved_termini = resolve_terminus_selections(cmd, working_directory_path, n_terminus, c_terminus)

        if select_termini and answers is None:
            # The AMBER ports, for example, patch termini through renamed terminal
            # residues and offer no menu, so run without -ter instead of failing.
            cmd.remove("-ter")
            run_checked_command(cmd, cwd=working_directory_path)
            status = ("Topology generated successfully. This force field offers no terminus "
                      "selection, so its own default termini were applied.")
        elif select_termini:
            process = subprocess.Popen(cmd, cwd=working_directory_path, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, stderr = process.communicate(input=answers)

            if process.returncode != 0:
                raise Exception(stderr)

            status = "Topology generated successfully. Termini: " + ", ".join(resolved_termini) + "."
        else:
            run_checked_command(cmd, cwd=working_directory_path)
            status = "Topology generated successfully."
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ligand_topology(working_directory_path: str, ligand_input_file_name: str,
                                ligand_output_file_name: str, ligand_charge: int,
                                ligand_charge_model: str,
                                ligand_force_field: str) -> tuple[list[str], str]:
    """Run acpype to parameterise the ligand with GAFF."""
    try:
        cmd = [
            "acpype",
            "-i", ligand_input_file_name,
            "-b", ligand_output_file_name,
            "-n", str(ligand_charge),
            "-c", ligand_charge_model,
            "-a", ligand_force_field
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd, cwd=working_directory_path)
        
        # Copy ligand structure and topology files to working directory
        ligand_dir = os.path.join(working_directory_path, f'{ligand_output_file_name}.acpype')
        shutil.copy2(os.path.join(ligand_dir, f'{ligand_output_file_name}_GMX.gro'), os.path.join(working_directory_path, f'{ligand_output_file_name}_GMX.gro'))
        shutil.copy2(os.path.join(ligand_dir, f'{ligand_output_file_name}_GMX.itp'), os.path.join(working_directory_path, f'{ligand_output_file_name}_GMX.itp'))

        status = "Topology generated successfully."
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_structures(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                        output_file: str) -> tuple[list[str], str]:
    """Combine protein and ligand coordinates into one complex structure."""
    try:
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        merge_protein_ligand_structures(protein_input_file_path, ligand_input_file_path, output_file_path)

        status = "Structure files merged successfully."
    except Exception as exc:
        status = "Error merging structure files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_topologies(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                        output_file: str) -> tuple[list[str], str]:
    """Combine the protein and ligand topologies into one complex topology."""
    try:
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        merge_protein_ligand_topologies(protein_input_file_path, ligand_input_file_path, output_file_path)

        status = "Topology files merged successfully."
    except Exception as exc:
        status = "Error merging topology files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_topology(working_directory_path: str, protein_input_file: str, ligand_input_file: str,
                      output_file: str) -> tuple[list[str], str]:
    """Combine the protein and ligand topologies into one complex topology."""
    try:
        protein_input_file_path = os.path.join(working_directory_path, protein_input_file)
        ligand_input_file_path = os.path.join(working_directory_path, ligand_input_file)
        output_file_path = os.path.join(working_directory_path, output_file)
        merge_protein_ligand_topologies(protein_input_file_path, ligand_input_file_path, output_file_path)

        status = "Topology files merged successfully."
    except Exception as exc:
        status = "Error merging topology files!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_simulation_box(working_directory_path: str, input_file_name: str, output_file_name: str,
                               box_type: str, distance: float) -> tuple[list[str], str]:
    """Run editconf to centre the solute in a box of the requested shape."""
    try:
        cmd = [
            "gmx", "editconf",
            "-f", os.path.join(working_directory_path, input_file_name),
            "-o", os.path.join(working_directory_path, output_file_name),
            "-c",
            "-d", str(distance),
            "-bt", box_type
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "Simulation box generated successfully."
    except Exception as exc:
        status = "Error generating simulation box!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_solvate_protein(working_directory_path: str, input_file_name: str, output_file_name: str,
                       input_topology_file_name: str, output_topology_file_name: str,
                       solvent_configuration: str) -> tuple[list[str], str]:
    """Run solvate to fill the box with the chosen solvent configuration."""
    try:
        shutil.copy2(os.path.join(working_directory_path, input_topology_file_name), os.path.join(working_directory_path, output_topology_file_name))

        cmd = [
            "gmx", "solvate",
            "-cp", os.path.join(working_directory_path, input_file_name),
            "-cs", solvent_configuration,
            "-o", os.path.join(working_directory_path, output_file_name),
            "-p", os.path.join(working_directory_path, output_topology_file_name)
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
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
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Ion addition parameter file generated successfully."
    except Exception as exc:
        status = "Error generating ion addition parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ions_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                              parameter_file_name: str, run_input_file_name: str, max_warnings: int) -> tuple[list[str], str]:
    """Run grompp to build the run input file that genion needs."""
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "Ion addition run input file generated successfully."
    except Exception as exc:
        status = "Error generating ion addition run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_add_ions_method_change(add_ions_method: str) -> tuple[GradioUpdate, ...]:
    """Show either the concentration slider or the explicit ion count sliders."""
    if add_ions_method == "Concentration":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:  # add_ions_method == "Number"
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def _find_sol_group(genion_cmd: Sequence[str], working_directory_path: str) -> str:
    """Detect the SOL group number genion offers, which depends on the topology.

    Group numbering is not fixed across force fields, so a probe run is parsed
    instead of assuming a well-known index."""
    tmp_gro = os.path.join(working_directory_path, ".probe_genion.gro")
    tmp_top = os.path.join(working_directory_path, ".probe_genion.top")

    probe_cmd = list(genion_cmd)
    probe_cmd[probe_cmd.index("-o") + 1] = tmp_gro
    top_idx = probe_cmd.index("-p") + 1
    shutil.copy2(probe_cmd[top_idx], tmp_top)
    probe_cmd[top_idx] = tmp_top

    try:
        probe = subprocess.Popen(probe_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr_probe = probe.communicate(input="0\n")
    finally:
        for f in [tmp_gro, tmp_top]:
            try:
                os.remove(f)
            except OSError:
                pass

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
        shutil.copy2(os.path.join(working_directory_path, input_topology_file_name), os.path.join(working_directory_path, output_topology_file_name))

        cmd = [
            "gmx", "genion",
            "-s", os.path.join(working_directory_path, run_input_file_name),
            "-o", os.path.join(working_directory_path, output_file_name),
            "-p", os.path.join(working_directory_path, output_topology_file_name),
            "-pname", cation_name,
            "-nname", anion_name,
        ]

        if neutralize:
            cmd.append("-neutral")

        if add_ion_method == "Concentration":
            cmd.extend(["-conc", str(concentration / 1000.0)])  # convert mM to M
        else:  # add_ion_method == "Number"
            cmd.extend(["-pq", str(cation_charge), "-np", str(number_of_cations), "-nq", str(anion_charge), "-nn", str(number_of_anions)])

        print(f"Running command: {' '.join(cmd)}")

        sol_group = _find_sol_group(cmd, working_directory_path)
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate(input=f"{sol_group}\n")

        if process.returncode != 0:
            raise Exception(stderr)

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
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Energy minimization parameter file generated successfully."
    except Exception as exc:
        status = "Error generating energy minimization parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_energy_minimization_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                             parameter_file_name: str, run_input_file_name: str, max_warnings: int) -> tuple[list[str], str]:
    """Run grompp to build the energy minimisation run input file."""
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "Energy minimization run input file generated successfully."
    except Exception as exc:
        status = "Error generating energy minimization run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_energy_minimization(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                               omp_threads: int, use_gpu: bool) -> tuple[list[str], str]:
    """Run mdrun for energy minimisation and wait for it to finish.

    use_gpu is deliberately ignored: GROMACS cannot run PME on the GPU during
    energy minimisation, so this step always stays on the CPU."""
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

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
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                           temperature: float, parameter_file_name: str,
                                           force_field: str) -> tuple[list[str], str]:
    """Write the restrained NVT equilibration MDP."""
    file_content = get_default_nvt_equilibration_mdp_file_content(time_scale_ps=time_scale, time_step_ps=time_step, temperature=temperature, with_ligand=True, force_field=force_field)
    try:
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "NVT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NVT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                           parameter_file_name: str, run_input_file_name: str, max_warnings: int) -> tuple[list[str], str]:
    """Run grompp to build the NVT run input file, with restraint references."""
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "NVT equilibration run input file generated successfully."
    except Exception as exc:
        status = "Error generating NVT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def watch_process(proc: subprocess.Popen[str], process_state: ProcessStateDict) -> None:
    """Clear the shared process state once the watched run exits."""
    proc.wait()  # wait until finished

    # If user already stopped it, do nothing
    with process_state["lock"]:
        if not process_state["running"]:
            return

        # Process finished naturally
        process_state["proc"] = None
        process_state["running"] = False

def sync_button_state(process_state: ProcessStateDict) -> GradioUpdate:
    """Keep a Run/Stop button label in step with the process state."""
    with process_state["lock"]:
        if process_state["running"]:
            return gr.update(value="Stop", variant="stop")
        else:
            return gr.update(value="Start", variant="primary")
    
def on_run_nvt_equilibration(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                             omp_threads: int, use_gpu: bool,
                             process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start NVT equilibration, or stop the run that is already in progress."""
    # ---------- STOP ----------
    with process_state["lock"]:
        was_running = process_state["running"]
        proc = process_state["proc"] if was_running else None
        process_state["proc"] = None
        process_state["running"] = False

    if was_running:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)

        status = "NVT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ] + get_mdrun_hardware_options(use_gpu, mpi_rank)

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True)

        with process_state["lock"]:
            process_state["proc"] = proc
            process_state["running"] = True

        threading.Thread(
            target=watch_process,
            args=(proc, process_state),
            daemon=True
        ).start()

        status = "NVT equilibration started."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        with process_state["lock"]:
            process_state["proc"] = None
            process_state["running"] = False

        status = f"Error during NVT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_generate_npt_equilibration_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                           temperature: float, pressure: float, parameter_file_name: str,
                                           force_field: str) -> tuple[list[str], str]:
    """Write the restrained NPT equilibration MDP."""
    file_content = get_default_npt_equilibration_mdp_file_content(time_scale_ps=time_scale, time_step_ps=time_step, temperature=temperature, pressure=pressure, with_ligand=True, force_field=force_field)
    try:
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "NPT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NPT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_npt_equilibration_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                           parameter_file_name: str, run_input_file_name: str, max_warnings: int) -> tuple[list[str], str]:
    """Run grompp to build the NPT run input file, with restraint references."""
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]
        
        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "NPT equilibration run input file generated successfully."
    except Exception as exc:
        status = "Error generating NPT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_npt_equilibration(working_directory_path: str, run_input_file_name: str, mpi_rank: int,
                             omp_threads: int, use_gpu: bool,
                             process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start NPT equilibration, or stop the run that is already in progress."""
    # ---------- STOP ----------
    with process_state["lock"]:
        was_running = process_state["running"]
        proc = process_state["proc"] if was_running else None
        process_state["proc"] = None
        process_state["running"] = False

    if was_running:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)

        status = "NPT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ] + get_mdrun_hardware_options(use_gpu, mpi_rank)

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True)

        with process_state["lock"]:
            process_state["proc"] = proc
            process_state["running"] = True

        threading.Thread(
            target=watch_process,
            args=(proc, process_state),
            daemon=True
        ).start()

        status = "NPT equilibration started."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        with process_state["lock"]:
            process_state["proc"] = None
            process_state["running"] = False

        status = f"Error during NPT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_toggle_nnpot(nnpot_active: bool) -> str:
    """Acknowledge the neural-network potential choice in the status line."""
    # The actual model is built by wrap_gmx_model.py when the production MD
    # parameter file is generated (it may need the input structure for atom
    # typing), so here we just acknowledge the choice.
    if not nnpot_active:
        return ""
    return ("<span style='color:green;'>Machine learning potential enabled. "
            "The selected model will be built when you generate the production MD parameter file.</span>")

def on_change_mdp_type(prod_md_mdp_type_radio: str) -> tuple[GradioUpdate, str]:
    """Switch the production MDP between an initial run and a continuation."""
    if prod_md_mdp_type_radio=="Initial":
        return gr.update(visible=True), "md_initial.mdp"
    else:
        return gr.update(visible=False), "md_continue.mdp"

def on_generate_prod_md_mdp_file(working_directory_path: str, time_scale: float, time_step: float,
                                 temperature: float, pressure: float, mdp_type: str, random_seed: int,
                                 parameter_file_name: str, nnpot_active: bool, nnpot_model_name: str,
                                 nnpot_input_group: str, force_field: str) -> tuple[list[str], str]:
    """Write the production MD MDP, building the neural potential if requested."""
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

    file_content = get_default_prod_md_mdp_file_content(time_scale_ps=time_scale*1000, time_step_ps=time_step, temperature=temperature, pressure=pressure, mdp_type=mdp_type, random_seed=random_seed, with_ligand=True, nnpot_active=nnpot_active, nnpot_modelfile_path=nnpot_modelfile_path, nnpot_input_group=nnpot_input_group, nnpot_model_name=nnpot_model_name, force_field=force_field)
    try:
        file_path = os.path.join(working_directory_path, parameter_file_name)
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Production MD parameter file generated successfully."
    except Exception as exc:
        status = "Error generating production MD parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_prod_md_tpr_file(working_directory_path: str, input_file_name: str, input_topology_file_name: str,
                                 parameter_file_name: str, run_input_file_name: str, max_warnings: int) -> tuple[list[str], str]:
    """Run grompp to build the production MD run input file."""
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", str(max_warnings),
            # Without -po this byproduct lands in the process working directory,
            # which is the repository root rather than the job directory.
            "-po", os.path.join(working_directory_path, "mdout.mdp")
        ]

        print(f"Running command: {' '.join(cmd)}")

        run_checked_command(cmd)
        status = "Production MD run input file generated successfully."
    except Exception as exc:
        status = "Error generating production MD run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_prod_md(working_directory_path: str, run_input_file_name: str, mpi_rank: int, omp_threads: int,
                   prod_md_nnpot_active: bool, use_gpu: bool,
                   process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Start production MD, or stop the run that is already in progress."""
    # ---------- STOP ----------
    with process_state["lock"]:
        was_running = process_state["running"]
        proc = process_state["proc"] if was_running else None
        process_state["proc"] = None
        process_state["running"] = False

    if was_running:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]
        if prod_md_nnpot_active:
            mpi_rank = 1

        cmd = [
            "gmx", "mdrun",
            "-deffnm", base_name,
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ]
        if use_gpu and not prod_md_nnpot_active:
            cmd.extend([
                "-nb", "gpu",
                "-pme", "gpu",
                "-bonded", "gpu",
                "-update", "gpu",
                "-pin", "on",
                "-dlb", "yes"
            ])
        elif not use_gpu:
            # Not merely "do not ask for the GPU": every task defaults to auto,
            # which picks a detected GPU, so the CPU has to be named.
            cmd.extend(get_cpu_only_mdrun_options())

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True)

        with process_state["lock"]:
            process_state["proc"] = proc
            process_state["running"] = True

        threading.Thread(
            target=watch_process,
            args=(proc, process_state),
            daemon=True
        ).start()

        status = "Production MD started."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        with process_state["lock"]:
            process_state["proc"] = None
            process_state["running"] = False

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_continue_prod_md(working_directory_path: str, run_input_file_name: str, checkpoint_file_name: str,
                        mpi_rank: int, omp_threads: int, prod_md_nnpot_active: bool, use_gpu: bool,
                        process_state: ProcessStateDict) -> tuple[Any, ...]:
    """Extend production MD from a checkpoint, or stop the running extension."""
    # ---------- STOP ----------
    with process_state["lock"]:
        was_running = process_state["running"]
        proc = process_state["proc"] if was_running else None
        process_state["proc"] = None
        process_state["running"] = False

    if was_running:
        # Shutting the run down happens outside the lock: waiting for mdrun to
        # write its checkpoint must not block the timer that polls this state.
        stop_process_gracefully(proc)

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]
        if prod_md_nnpot_active:
            mpi_rank = 1

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
            cmd.extend([
                "-nb", "gpu",
                "-pme", "gpu",
                "-bonded", "gpu",
                "-update", "gpu",
                "-pin", "on",
                "-dlb", "yes"
            ])
        elif not use_gpu:
            # Not merely "do not ask for the GPU": every task defaults to auto,
            # which picks a detected GPU, so the CPU has to be named.
            cmd.extend(get_cpu_only_mdrun_options())

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd=working_directory_path, text=True)

        with process_state["lock"]:
            process_state["proc"] = proc
            process_state["running"] = True

        threading.Thread(
            target=watch_process,
            args=(proc, process_state),
            daemon=True
        ).start()

        status = "Production MD started."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        with process_state["lock"]:
            process_state["proc"] = None
            process_state["running"] = False

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")
    
def on_make_molecule_whole(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                           output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -pbc whole to repair molecules broken across the box edge."""
    try:
        cmd = [
            "gmx", "trjconv",
            "-s", os.path.join(working_directory_path, run_input_file_name),
            "-f", os.path.join(working_directory_path, input_traj_file_name),
            "-o", os.path.join(working_directory_path, output_traj_file_name),
            "-pbc", "whole"
        ]

        print(f"Running command: {' '.join(cmd)}")

        # trjconv requires user input to select a group; we will provide "0" for "System"
        run_checked_command(cmd, stdin_input="0\n")
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_center_protein(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                      output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -pbc mol -center to keep the solute in the middle of the box."""
    try:
        cmd = [
            "gmx", "trjconv",
            "-s", os.path.join(working_directory_path, run_input_file_name),
            "-f", os.path.join(working_directory_path, input_traj_file_name),
            "-o", os.path.join(working_directory_path, output_traj_file_name),
            "-pbc", "mol",
            "-center"
        ]

        print(f"Running command: {' '.join(cmd)}")

        # trjconv requires user input to select a group; we will provide "1" for "Protein", then "0" for "System"
        run_checked_command(cmd, stdin_input="1\n0\n")
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_fit_backbone(working_directory_path: str, run_input_file_name: str, input_traj_file_name: str,
                    output_traj_file_name: str) -> tuple[list[str], str]:
    """Run trjconv -fit rot+trans to remove overall rotation and translation."""
    try:
        cmd = [
            "gmx", "trjconv",
            "-s", os.path.join(working_directory_path, run_input_file_name),
            "-f", os.path.join(working_directory_path, input_traj_file_name),
            "-o", os.path.join(working_directory_path, output_traj_file_name),
            "-fit", "rot+trans"
        ]

        print(f"Running command: {' '.join(cmd)}")

        # trjconv requires user input to select a group; we will provide "4" for "Backbone", then "0" for "System"
        run_checked_command(cmd, stdin_input="4\n0\n")
        
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

def on_analyze_rmsd(working_directory_path: str, structure_file_name: str,
                    input_traj_file_name: str) -> tuple[Any, ...]:
    """Backbone RMSD of the protein and of the ligand against the first frame.

    Both series stay on one plot: they are two readings of the same measurement
    and are compared against each other.
    """
    try:
        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        _require_ligand(universe)

        protein_rmsd = rms.RMSD(
            universe,
            select="protein and backbone",
            groupselections=["protein"],
            ref_frame=0
        ).run()

        ligand_rmsd = rms.RMSD(
            universe,
            select=f"resname {LIGAND_RESNAME}",
            groupselections=[f"resname {LIGAND_RESNAME}"],
            ref_frame=0,
            rmsd_kwargs={"center": True, "superposition": True}
        ).run()

        frame = pd.DataFrame({"Time (ns)": protein_rmsd.results.rmsd[:, 1] / 1000,
                              "Protein RMSD (Å)": protein_rmsd.results.rmsd[:, 2],
                              "Ligand RMSD (Å)": ligand_rmsd.results.rmsd[:, 2]})
        figure = make_line_figure(frame, "Time (ns)", ylabel="RMSD (Å)", title="RMSD vs Time")
        status = "RMSD calculated successfully."
    except Exception as exc:
        status = "Error calculating RMSD!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_min_distance(working_directory_path: str, structure_file_name: str,
                            input_traj_file_name: str) -> tuple[Any, ...]:
    """Closest approach between any protein atom and any ligand atom, per frame.

    Complements the centre of mass distance: two molecules in contact can still
    have their centres far apart, so this is what tells you whether the ligand is
    actually touching the protein rather than merely near it.
    """
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
            pairwise = distances.distance_array(protein_selector.positions,
                                                ligand_selector.positions,
                                                box=universe.dimensions)
            times_ns.append(timestep.time / 1000)
            minimum_distances.append(float(pairwise.min()))

        frame = pd.DataFrame({"Time (ns)": times_ns,
                              "Minimum distance (Å)": minimum_distances})
        figure = make_line_figure(frame, "Time (ns)", ylabel="Minimum distance (Å)",
                                  title="Protein-ligand minimum distance", mean_line=True)
        status = "Minimum distance calculated successfully."
    except Exception as exc:
        status = "Error calculating minimum distance!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_com_distance(working_directory_path: str, structure_file_name: str,
                            input_traj_file_name: str) -> tuple[Any, ...]:
    """Distance between the protein and ligand centres of mass, frame by frame."""
    try:
        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        protein_selector = universe.select_atoms("protein")
        ligand_selector = _require_ligand(universe)

        # The time axis is built here rather than borrowed from an RMSD result, so
        # this analysis stands on its own now that it has its own button.
        times_ns = []
        distances = []
        for timestep in universe.trajectory:
            times_ns.append(timestep.time / 1000)
            distances.append(float(np.linalg.norm(
                protein_selector.center_of_mass() - ligand_selector.center_of_mass())))

        frame = pd.DataFrame({"Time (ns)": times_ns,
                              "Center of mass distance (Å)": distances})
        figure = make_line_figure(frame, "Time (ns)", ylabel="Center of mass distance (Å)",
                                  title="Protein-ligand centre of mass distance")
        status = "Center of mass distance calculated successfully."
    except Exception as exc:
        status = "Error calculating center of mass distance!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def on_analyze_rmsf(working_directory_path: str, structure_file_name: str,
                    input_traj_file_name: str) -> tuple[Any, ...]:
    """Per-residue fluctuation of the C-alpha atoms over the whole trajectory."""
    try:
        universe = mda.Universe(os.path.join(working_directory_path, structure_file_name),
                                os.path.join(working_directory_path, input_traj_file_name))
        ca_selector = universe.select_atoms("protein and name CA")
        if ca_selector.n_atoms == 0:
            raise Exception("No C-alpha atoms found. Is this a protein structure?")

        ca_rmsf = rms.RMSF(ca_selector).run().results.rmsf

        frame = pd.DataFrame({"Residue Index": ca_selector.resids, "Cα RMSF (Å)": ca_rmsf})
        figure = make_line_figure(frame, "Residue Index", ylabel="RMSF (Å)",
                                  title="Cα RMSF per Residue", mean_line=True)
        status = "RMSF calculated successfully."
    except Exception as exc:
        status = "Error calculating RMSF!\n" + str(exc)
        return None, None, "<span style='color:red;'>" + status + "</span>"

    return frame, figure, "<span style='color:green;'>" + status + "</span>"

def _selection_error(exc: Exception, run_input_file_name: str,
                     working_directory_path: str) -> str:
    """A gmx failure message, with the structure's own residue groups appended
    when the cause was a selection that matched nothing."""
    message = str(exc)
    if "never matches any atoms" not in message and "Invalid selection" not in message:
        return message

    hint = describe_selection_candidates(run_input_file_name, working_directory_path)
    return f"{message}\n\n{hint}" if hint else message

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

        first = int(first_eigenvector)
        second = int(second_eigenvector)
        if second <= first:
            raise Exception("The second eigenvector must be higher than the first.")

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
# How many residues the contribution chart shows; the exported table keeps all.
MMPBSA_DECOMPOSITION_RESIDUES_SHOWN: int = 15

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

        with open(os.path.join(working_directory_path, mmpbsa_input_file_name), "w") as file:
            file.write(file_content)
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
    with process_state["lock"]:
        was_running = process_state["running"]
        proc = process_state["proc"] if was_running else None
        process_state["proc"] = None
        process_state["running"] = False

    if was_running:
        # Outside the lock: waiting on the shutdown must not block the timer that
        # polls this state.
        stop_process_gracefully(proc)

        status = "MM-PBSA stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        executable = get_gmx_mmpbsa_executable()
        if executable is None:
            # "or" guards the case where the two helpers disagree: raising
            # Exception(None) would show the user the word "None".
            raise Exception(get_gmx_mmpbsa_unavailable_reason()
                            or "gmx_MMPBSA was not found. See the Readme for how to install it.")

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
                                    env=get_gmx_mmpbsa_environment(executable))

        with process_state["lock"]:
            process_state["proc"] = proc
            process_state["running"] = True

        threading.Thread(
            target=watch_process,
            args=(proc, process_state),
            daemon=True
        ).start()

        status = (f"MM-PBSA started. This can take a long time; load the results when "
                  f"the button returns to Start. Progress and any error are written to "
                  f"{MMPBSA_LOG_FILE_NAME}, which opens in the text viewer.")

        return get_files_in_working_directory(working_directory_path), f"<span style='color:orange;'>{status}</span>", process_state, gr.update(value="Stop", variant="stop")

    except Exception as exc:
        with process_state["lock"]:
            process_state["proc"] = None
            process_state["running"] = False

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
        results_file_path = os.path.join(working_directory_path, mmpbsa_results_file_name)
        if not os.path.exists(results_file_path):
            # Runs started before the move out of the scratch subdirectory left
            # their results there, so those stay readable.
            legacy_path = os.path.join(working_directory_path, MMPBSA_SUBDIRECTORY,
                                       mmpbsa_results_file_name)
            if os.path.exists(legacy_path):
                results_file_path = legacy_path
            else:
                raise Exception(f"{mmpbsa_results_file_name} was not found. Has the run "
                                f"finished? {MMPBSA_LOG_FILE_NAME} shows how far it got.")

        frame = parse_mmpbsa_results(results_file_path)
        # Error bars use the plain per-frame SD rather than SD(Prop.): the
        # propagated one describes the components, not the spread of the delta.
        figure = make_bar_figure(frame, "Term", "Average (kcal/mol)", "SD",
                                 ylabel="ΔG (kcal/mol)", title="MM-PBSA energy decomposition")

        # Copy the results out of the scratch directory so they show in the file
        # table and can be opened in the text viewer.
        if os.path.dirname(results_file_path) != os.path.abspath(working_directory_path):
            shutil.copy2(results_file_path,
                         os.path.join(working_directory_path, mmpbsa_results_file_name))

        results_directory_path = os.path.dirname(results_file_path)
        # The per-frame and per-residue files sit beside the summary and are only
        # written when the run asked for them, so each is optional.
        histogram_figure, missing = _load_binding_energy_histogram(results_directory_path)
        series_figure, series_note = _load_binding_energy_series(
            results_directory_path, working_directory_path, structure_file_name,
            input_traj_file_name, mmpbsa_input_file_name)
        decomposition, decomposition_figure, decomposition_missing = \
            _load_residue_decomposition(results_directory_path)

        status = "MM-PBSA results loaded successfully."
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

def _load_binding_energy_histogram(results_directory_path: str) -> tuple[Any, str]:
    """The spread of the binding energy over the frames, if -eo was written."""
    per_frame_path = os.path.join(results_directory_path, MMPBSA_PER_FRAME_FILE_NAME)
    if not os.path.exists(per_frame_path):
        return None, (f"No {MMPBSA_PER_FRAME_FILE_NAME}, so there is no per-frame "
                      f"distribution to plot.")

    per_frame = parse_mmpbsa_per_frame(per_frame_path)
    figure = make_histogram_figure(per_frame["TOTAL"], bins=30,
                                   xlabel="ΔG binding (kcal/mol)",
                                   title=f"Binding energy over {len(per_frame)} frames")
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
    # Only the residues that matter: a long tail of near-zero contributions
    # would leave the significant ones unreadable.
    strongest = decomposition.reindex(
        decomposition["TOTAL"].abs().sort_values(ascending=False).index
    ).head(MMPBSA_DECOMPOSITION_RESIDUES_SHOWN).sort_values("TOTAL")
    colours, legend = mmpbsa_residue_colours(strongest["Residue"])
    figure = make_bar_figure(strongest, "Residue", "TOTAL", "TOTAL SD",
                             ylabel="ΔG contribution (kcal/mol)",
                             title=f"Strongest {len(strongest)} residue contributions",
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
    note = ""
    times_ns: list[float] = []
    input_file_path = os.path.join(working_directory_path, mmpbsa_input_file_name)
    try:
        start_frame, interval = read_mmpbsa_frame_selection(input_file_path)
        times_ns = get_trajectory_frame_times_ns(
            os.path.join(working_directory_path, structure_file_name),
            os.path.join(working_directory_path, input_traj_file_name),
            start_frame, interval, len(per_frame))
    except Exception as exc:
        note = (f"Binding energy is plotted against frame number: the times could "
                f"not be read from {input_traj_file_name} ({exc}).")

    if len(times_ns) == len(per_frame):
        frame = pd.DataFrame({"Time (ns)": times_ns,
                              "ΔG binding (kcal/mol)": per_frame["TOTAL"].to_numpy()})
        x_column = "Time (ns)"
    else:
        if not note:
            note = (f"Binding energy is plotted against frame number: the trajectory "
                    f"holds fewer frames than the run used.")
        frame = pd.DataFrame({"Frame": per_frame["Frame #"].to_numpy(),
                              "ΔG binding (kcal/mol)": per_frame["TOTAL"].to_numpy()})
        x_column = "Frame"

    figure = make_line_figure(frame, x_column, ylabel="ΔG binding (kcal/mol)",
                              title="Binding energy over the trajectory", mean_line=True)
    return figure, note

def on_export_df(working_directory_path: str, df: pd.DataFrame, file_name: str) -> tuple[list[str], str]:
    """Write an analysis table to CSV inside the job directory."""
    if df is None:
        # One export button per analysis now, so exporting before running the
        # matching analysis is an easy mistake to make.
        return get_files_in_working_directory(working_directory_path), \
            "<span style='color:red;'>Run the analysis before exporting its results.</span>"

    try:
        df.to_csv(os.path.join(working_directory_path, file_name), index=False)
        status = f"File exported: {file_name}"
    except Exception as exc:
        status = "Error exporting file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"  
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

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
                        max_warns_slider = gr.Slider(label="Max Warnings", minimum=0, maximum=10, value=5, step=1)
                        use_gpu = gr.Checkbox(label="Use GPU", value=True)
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
                                    protein_force_field_dropdown = gr.Dropdown(label="Force Field", choices=["AMBER94", "AMBER96", "AMBER99", "AMBER99SB", "AMBER99SB-ILDN", "AMBER03", "AMBERGS", "AMBER14SB", "AMBER19SB",
                                                                                                    "CHARMM27", "CHARMM36", "GROMOS43A1", "GROMOS43A2", "GROMOS45A3", "GROMOS53A5", "GROMOS53A6", "GROMOS54A7", ("OPLS-AA", "OPLSAA")], value="AMBER99SB-ILDN", allow_custom_value=True)
                                    water_model_dropdown = gr.Dropdown(label="Water Model", choices=["SELECT", "NONE", "OPC", "OPC3", "SPC", "SPCE", "TIP3P", "TIP4P", ("TIP4P-Ew", "TIP4PEW"), "TIP5P", "TIPS3P"], value="TIP3P")
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
                            distance_slider = gr.Slider(label="Distance to Box Edge (nm)", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                            generate_box_button = gr.Button(value="Generate Simulation Box")
                with gr.Accordion(label="Solvation", open=False):
                    with gr.Row():
                        with gr.Column():
                            solvation_input_file_name_dropdown = gr.Dropdown(label="Input File Name", choices=[], value=None)
                            solvation_output_file_name_textbox = gr.Textbox(label="Output File Name", value="solvated_complex.gro")
                            solvation_input_topology_file_name_dropdown = gr.Dropdown(label="Input Topology File Name", choices=[], value=None)
                            solvation_output_topology_file_name_textbox = gr.Textbox(label="Output Topology File Name", value="solvated_topology.top")
                        with gr.Column():
                            solvent_configuration_dropdown = gr.Dropdown(label="Solvent Configuration", choices=["spc216.gro", "tip4p.gro", "tip5p.gro"], value="spc216.gro")
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
                                    cation_charge_slider = gr.Slider(label="Cation Charge", minimum=1, maximum=3, value=1, step=1, visible=False)
                                    anion_charge_slider = gr.Slider(label="Anion Charge", minimum=-3, maximum=-1, value=-1, step=1, visible=False)
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
                                    nvt_time_step_slider = gr.Slider(label="Time Step (ps)", minimum=0.001, maximum=0.005, value=0.002, step=0.001)
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
                                    nvt_equilibration_timer = gr.Timer(1.0)
                with gr.Accordion(label="NPT Equilibration", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for NPT equilibration***")
                            with gr.Row():
                                with gr.Column():
                                    npt_time_scale_slider = gr.Slider(label="NPT Equilibration Time (ps)", minimum=100, maximum=5000, value=1000, step=100)
                                    npt_time_step_slider = gr.Slider(label="Time Step (ps)", minimum=0.001, maximum=0.005, value=0.002, step=0.001)
                                    npt_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                    npt_pressure_slider = gr.Slider(label="Pressure (atm)", minimum=0.1, maximum=10, value=1, step=0.1)
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
                                    npt_equilibration_timer = gr.Timer(1.0)
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
                                            prod_md_time_step_slider = gr.Slider(label="Time Step (ps)", minimum=0.001, maximum=0.005, value=0.002, step=0.001)
                                            prod_md_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                            prod_md_pressure_slider = gr.Slider(label="Pressure (atm)", minimum=0.1, maximum=10, value=1, step=0.1)
                                    with gr.Row():
                                        prod_md_nnpot_active_checkbox = gr.Checkbox(label="Use Machine Learning Potential (NNPot)", value=False)
                                        prod_md_nnpot_model_dropdown = gr.Dropdown(label="Model", choices=["ani1x", "ani2x", "ani2x-emle", "mace-small", "mace-medium", "mace-large", "aimnet2"], value="ani2x")
                                        prod_md_nnpot_input_group_textbox = gr.Textbox(label="NNPot Input Group", value="Protein")
                                with gr.Column():
                                    prod_md_mdp_type_radio = gr.Radio(label="Initial or continuation", choices=["Initial", "Continuation"], value="Initial")
                                    prod_md_random_seed_textbox = gr.Textbox(label="Random seed", value="0")
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
                                    prod_md_initial_timer = gr.Timer(1.0)
                                with gr.Column():
                                    gr.Markdown("*Run from a checkpoint*")
                                    prod_md_continuation_process_state = gr.State(ProcessStateDict())
                                    checkpoint_file_dropdown = gr.Dropdown(label="Checkpoint File Name", choices=[], value=None)
                                    continue_prod_md_button = gr.Button(value="Continue production MD simulation")
                                    prod_md_continuation_timer = gr.Timer(1.0)
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
                                    mmpbsa_timer = gr.Timer(1.0)
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
    working_directory_dropdown.change(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, clean_working_directory_button, protein_structure_file, ligand_structure_file])
    open_working_directory_button.click(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, clean_working_directory_button, protein_structure_file, ligand_structure_file])
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
    generate_ligand_topology_button.click(on_generate_ligand_topology, [working_directory_path_state, ligand_topology_input_file_name_dropdown, ligand_output_file_name_textbox, ligand_charge_slider, ligand_charge_model_dropdown, ligand_force_field_dropdown], [working_directory_file_list_state, status_markdown])

    # Merge structure and topology interaction
    merge_structures_button.click(on_merge_structures, [working_directory_path_state, merge_structures_protein_input_file_name_dropdown, merge_structures_ligand_input_file_name_dropdown, merge_structures_output_file_name_textbox], [working_directory_file_list_state, status_markdown])
    merge_topologies_button.click(on_merge_topologies, [working_directory_path_state, merge_topologies_protein_input_file_name_dropdown, merge_topologies_ligand_input_file_name_dropdown, merge_topologies_output_file_name_textbox], [working_directory_file_list_state, status_markdown])

    # Generate simulation box interaction
    generate_box_button.click(on_generate_simulation_box, [working_directory_path_state, box_input_file_name_dropdown, box_output_file_name_textbox, box_type_dropdown, distance_slider], [working_directory_file_list_state, status_markdown])

    # Solvation interaction
    solvate_button.click(on_solvate_protein, [working_directory_path_state, solvation_input_file_name_dropdown, solvation_output_file_name_textbox, solvation_input_topology_file_name_dropdown, solvation_output_topology_file_name_textbox, solvent_configuration_dropdown], [working_directory_file_list_state, status_markdown])

    # Generate ions interaction
    generate_ions_parameter_file_button.click(on_generate_ions_mdp_file, [working_directory_path_state, generate_ions_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    generate_ions_run_input_file_button.click(on_generate_ions_tpr_file, [working_directory_path_state, generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_name_textbox, max_warns_slider], [working_directory_file_list_state, status_markdown])
    add_ion_method_radio.change(on_add_ions_method_change, add_ion_method_radio, [concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider])
    add_ions_button.click(on_add_ions, [working_directory_path_state, generate_ions_run_input_file_dropdown, generate_ions_output_file_name_textbox, generate_ions_input_topology_file_name_dropdown, generate_ions_output_topology_file_name_textbox, cation_name_textbox, anion_name_textbox, add_ion_method_radio, concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider, netralize_checkbox], [working_directory_file_list_state, status_markdown])
    
    # Energy minimization interaction
    energy_minimization_parameter_file_button.click(on_generate_energy_minimization_mdp_file, [working_directory_path_state, energy_minimization_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    energy_minimization_run_input_file_button.click(on_generate_energy_minimization_tpr_file, [working_directory_path_state, energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_name_textbox, max_warns_slider], [working_directory_file_list_state, status_markdown])
    run_energy_minimization_button.click(on_run_energy_minimization, [working_directory_path_state, energy_minimization_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu], [working_directory_file_list_state, status_markdown])

    # NVT equilibration interaction
    nvt_equilibration_parameter_file_button.click(on_generate_nvt_equilibration_mdp_file, [working_directory_path_state, nvt_time_scale_slider, nvt_time_step_slider, nvt_temperature_slider, nvt_equilibration_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    nvt_equilibration_run_input_file_button.click(on_generate_nvt_equilibration_tpr_file, [working_directory_path_state, nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_name_textbox, max_warns_slider], [working_directory_file_list_state, status_markdown])
    run_nvt_equilibration_button.click(on_run_nvt_equilibration, [working_directory_path_state, nvt_equilibration_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, nvt_process_state], [working_directory_file_list_state, status_markdown, nvt_process_state, run_nvt_equilibration_button])
    nvt_equilibration_timer.tick(sync_button_state, nvt_process_state, run_nvt_equilibration_button)

    # NPT equilibration interaction
    npt_equilibration_parameter_file_button.click(on_generate_npt_equilibration_mdp_file, [working_directory_path_state, npt_time_scale_slider, npt_time_step_slider, npt_temperature_slider, npt_pressure_slider, npt_equilibration_parameter_file_name_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    npt_equilibration_run_input_file_button.click(on_generate_npt_equilibration_tpr_file, [working_directory_path_state, npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_name_textbox, max_warns_slider], [working_directory_file_list_state, status_markdown])
    run_npt_equilibration_button.click(on_run_npt_equilibration, [working_directory_path_state, npt_equilibration_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, npt_process_state], [working_directory_file_list_state, status_markdown, npt_process_state, run_npt_equilibration_button])
    npt_equilibration_timer.tick(sync_button_state, npt_process_state, run_npt_equilibration_button)

    # Production MD interaction
    prod_md_mdp_type_radio.change(on_change_mdp_type, prod_md_mdp_type_radio, [prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox])
    prod_md_nnpot_active_checkbox.change(on_toggle_nnpot, prod_md_nnpot_active_checkbox, status_markdown)
    prod_md_parameter_file_button.click(on_generate_prod_md_mdp_file, [working_directory_path_state, prod_md_time_scale_slider, prod_md_time_step_slider, prod_md_temperature_slider, prod_md_pressure_slider, prod_md_mdp_type_radio, prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox, prod_md_nnpot_active_checkbox, prod_md_nnpot_model_dropdown, prod_md_nnpot_input_group_textbox, protein_force_field_dropdown], [working_directory_file_list_state, status_markdown])
    prod_md_run_input_file_button.click(on_generate_prod_md_tpr_file, [working_directory_path_state, prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_name_textbox, max_warns_slider], [working_directory_file_list_state, status_markdown])
    run_prod_md_button.click(on_run_prod_md, [working_directory_path_state, prod_md_run_input_file_dropdown, mpi_rank_slider, omp_threads_slider, prod_md_nnpot_active_checkbox, use_gpu, prod_md_initial_process_state], [working_directory_file_list_state, status_markdown, prod_md_initial_process_state, run_prod_md_button])
    prod_md_initial_timer.tick(sync_button_state, prod_md_initial_process_state, run_prod_md_button)
    continue_prod_md_button.click(on_continue_prod_md, [working_directory_path_state, prod_md_run_input_file_dropdown, checkpoint_file_dropdown, mpi_rank_slider, omp_threads_slider, prod_md_nnpot_active_checkbox, use_gpu, prod_md_continuation_process_state], [working_directory_file_list_state, status_markdown, prod_md_continuation_process_state, continue_prod_md_button])
    prod_md_continuation_timer.tick(sync_button_state, prod_md_continuation_process_state, continue_prod_md_button)

    # Fix trajectory interaction
    make_mol_whole_button.click(on_make_molecule_whole, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, make_mol_whole_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    center_protein_button.click(on_center_protein, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, center_protein_input_traj_file_name_dropdown, center_protein_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    fit_backbone_button.click(on_fit_backbone, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown, fit_backbone_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])

    # Analysis
    rmsd_analyze_button.click(on_analyze_rmsd, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [rmsd_df_state, rmsd_plot, status_markdown])
    min_dist_analyze_button.click(on_analyze_min_distance, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [min_dist_df_state, min_dist_plot, status_markdown])
    com_dist_analyze_button.click(on_analyze_com_distance, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [com_dist_df_state, com_dist_plot, status_markdown])
    ca_rmsf_analyze_button.click(on_analyze_rmsf, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [ca_rmsf_df_state, ca_rmsf_plot, status_markdown])
    sasa_analyze_button.click(on_analyze_sasa, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, sasa_surface_selection_textbox, sasa_output_selection_textbox, sasa_probe_radius_slider, sasa_output_file_name_textbox, sasa_residue_output_file_name_textbox], [working_directory_file_list_state, sasa_df_state, sasa_plot, sasa_residue_df_state, sasa_residue_plot, status_markdown])
    mmpbsa_input_file_button.click(on_generate_mmpbsa_input_file, [working_directory_path_state, mmpbsa_input_file_name_textbox, mmpbsa_start_frame_textbox, mmpbsa_end_frame_textbox, mmpbsa_interval_slider, mmpbsa_salt_concentration_slider, mmpbsa_temperature_slider, mmpbsa_method_checkboxgroup, mmpbsa_decomposition_checkbox, mmpbsa_decomposition_scheme_dropdown, mmpbsa_print_residues_textbox], [working_directory_file_list_state, status_markdown])
    run_mmpbsa_button.click(on_run_mmpbsa, [working_directory_path_state, analysis_run_input_file_name_dropdown, analysis_input_traj_file_name_dropdown, mmpbsa_input_topology_file_name_dropdown, mmpbsa_input_file_name_textbox, mmpbsa_index_file_name_textbox, mmpbsa_receptor_selection_textbox, mmpbsa_ligand_selection_textbox, mmpbsa_processes_slider, mmpbsa_process_state], [working_directory_file_list_state, status_markdown, mmpbsa_process_state, run_mmpbsa_button])
    mmpbsa_timer.tick(sync_button_state, mmpbsa_process_state, run_mmpbsa_button)
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
