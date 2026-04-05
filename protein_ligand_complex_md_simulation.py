import os
import time
import threading
import psutil
import shutil
import subprocess
import numpy as np
import pandas as pd
import gradio as gr
import parmed as pmd
import nglview
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
from utils import *

def get_working_directories():
    base_path = "./data/"
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def get_files_in_working_directory(working_directory_path):
    files = [f for f in os.listdir(working_directory_path) if not (f.startswith('#') or f.endswith("Zone.Identifier") or os.path.isdir(os.path.join(working_directory_path, f)))]
    return files

def on_open_working_directory(working_directory):
    if working_directory is None or working_directory.strip() == "":
        gr.Warning("Please specify a working directory.")
        return None, None, None, None, None, None
    
    working_directory_path = os.path.join("./data/", working_directory)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_files_in_working_directory(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory), working_directory_path, files, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

def on_file_list_change(working_directory_path,
                        protein_structure_file_name, ligand_structure_file_name, protein_topology_output_file_name, ligand_output_file_name, protein_topology_output_topology_file_name,
                        merge_structures_output_file_name, box_output_file_name, merge_topologies_output_file_name,
                        solvation_output_file_name, solvation_output_topology_file_name,
                        generate_ions_parameter_file_name, generate_ions_run_input_file_name, generate_ions_output_file_name, generate_ions_output_topology_file_name,
                        energy_minimization_parameter_file_name, energy_minimization_run_input_file_name,
                        nvt_equilibration_parameter_file_name, nvt_equilibration_run_input_file_name,
                        npt_equilibration_parameter_file_name, npt_equilibration_run_input_file_name,
                        prod_md_parameter_file_name, prod_md_run_input_file_name,
                        make_mol_whole_output_traj_file_name, center_protein_output_traj_file_name, fit_backbone_output_traj_file_name):
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
        elif f.endswith('.csv'):
            file_type = "Data File"
        else:
            file_type = "Other File"
        modified_time = time.ctime(os.path.getmtime(file_path))
        file_info.append([f, file_type, modified_time])
        file_info.sort(key=lambda x: x[2].lower(), reverse=True) # Sort by modified time descending
    file_df = pd.DataFrame(file_info, columns=["File", "Type", "Modified"])

    # Filter structure and text files
    structure_files = [f for f in files if f.endswith('.pdb') or f.endswith('.gro')]
    topology_files = [f for f in files if f.endswith('.top') or f.endswith('.itp')]
    parameter_files = [f for f in files if f.endswith('.mdp')]
    run_input_files = [f for f in files if f.endswith('.tpr')]
    checkpoint_files = [f for f in files if f.endswith('.cpt')]
    trajectory_files = [f for f in files if f.endswith('.xtc')]

    # Update protein topology input file name dropdown
    if protein_structure_file_name in structure_files:
        protein_topology_input_file_name = protein_structure_file_name
    else:
        protein_topology_input_file_name = structure_files[0] if structure_files else None

    # Update ligand topology input file name dropdown
    if ligand_structure_file_name in structure_files:
        ligand_topology_input_file_name = ligand_structure_file_name
    else:
        ligand_topology_input_file_name = structure_files[0] if structure_files else None
    
    # Update merge structure protein input file name dropdown
    if protein_topology_output_file_name in structure_files:
        merge_structure_protein_input_file_name = protein_topology_output_file_name
    else:
        merge_structure_protein_input_file_name = structure_files[0] if structure_files else None
    
    # Update merge structure ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.gro" in structure_files:
        merge_structure_ligand_input_file_name = f"{ligand_output_file_name}_GMX.gro"
    else:
        merge_structure_ligand_input_file_name = structure_files[0] if structure_files else None

    # Update merge topology protein input file name dropdown
    if protein_topology_output_topology_file_name in topology_files:
        merge_topology_protein_input_file_name = protein_topology_output_topology_file_name
    else:
        merge_topology_protein_input_file_name = topology_files[0] if topology_files else None
    
    # Update merge topology ligand input file name dropdown
    if f"{ligand_output_file_name}_GMX.itp" in topology_files:
        merge_topology_ligand_input_file_name = f"{ligand_output_file_name}_GMX.itp"
    else:
        merge_topology_ligand_input_file_name = topology_files[0] if topology_files else None

    # Update box input file name dropdown
    if merge_structures_output_file_name in structure_files:
        box_input_file_name = merge_structures_output_file_name
    else:
        box_input_file_name = structure_files[0] if structure_files else None

    # Update solvation input file dropdown
    if box_output_file_name in structure_files:
        solvation_input_file_name = box_output_file_name
    else:
        solvation_input_file_name = structure_files[0] if structure_files else None

    # Update solvation input topology file dropdown
    if merge_topologies_output_file_name in topology_files:
        solvation_input_topology_file_name = merge_topologies_output_file_name
    else:
        solvation_input_topology_file_name = topology_files[0] if topology_files else None

    # Update generate ions input file dropdown
    if solvation_output_file_name in structure_files:
        generate_ions_input_file_name = solvation_output_file_name
    else:
        generate_ions_input_file_name = structure_files[0] if structure_files else None

    # Update generate ions input topology file dropdown
    if solvation_output_topology_file_name in topology_files:
        generate_ions_input_topology_file_name = solvation_output_topology_file_name
    else:
        generate_ions_input_topology_file_name = topology_files[0] if topology_files else None

    # Update generate ions parameter file dropdown
    if generate_ions_parameter_file_name in parameter_files:
        generate_ions_parameter_file_name = generate_ions_parameter_file_name
    else:
        generate_ions_parameter_file_name = parameter_files[0] if parameter_files else None

    # Update generate ions run input file dropdown
    if generate_ions_run_input_file_name in run_input_files:
        generate_ions_run_input_file_name = generate_ions_run_input_file_name
    else:
        generate_ions_run_input_file_name = run_input_files[0] if run_input_files else None

    # Update energy minimization input file dropdown
    if generate_ions_output_file_name in structure_files:
        energy_minimization_input_file_name = generate_ions_output_file_name
    else:
        energy_minimization_input_file_name = structure_files[0] if structure_files else None

    # Update energy minimization run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        energy_minimization_input_topology_file_name = generate_ions_output_topology_file_name
    else:
        energy_minimization_input_topology_file_name = topology_files[0] if topology_files else None

    # Update energy minimization parameter file dropdown
    if energy_minimization_parameter_file_name in parameter_files:
        energy_minimization_parameter_file_name = energy_minimization_parameter_file_name
    else:
        energy_minimization_parameter_file_name = parameter_files[0] if parameter_files else None

    # Update nvt equilibration input file dropdown
    if energy_minimization_run_input_file_name in run_input_files:
        nvt_equilibration_input_file_name = f"{energy_minimization_run_input_file_name.split('.')[0]}.gro"
    else:
        nvt_equilibration_input_file_name = structure_files[0] if structure_files else None

    # Update nvt equilibration run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        nvt_equilibration_input_topology_file_name = generate_ions_output_topology_file_name
    else:
        nvt_equilibration_input_topology_file_name = topology_files[0] if topology_files else None

    # Update nvt equilibration parameter file dropdown
    if nvt_equilibration_parameter_file_name in parameter_files:
        nvt_equilibration_parameter_file_name = nvt_equilibration_parameter_file_name
    else:
        nvt_equilibration_parameter_file_name = parameter_files[0] if parameter_files else None

    # Update npt equilibration input file dropdown
    if nvt_equilibration_run_input_file_name in run_input_files:
        npt_equilibration_input_file_name = f"{nvt_equilibration_run_input_file_name.split('.')[0]}.gro"
    else:
        npt_equilibration_input_file_name = structure_files[0] if structure_files else None
    
    # Update npt equilibration run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        npt_equilibration_input_topology_file_name = generate_ions_output_topology_file_name
    else:
        npt_equilibration_input_topology_file_name = topology_files[0] if topology_files else None
    
    # Update npt equilibration parameter file dropdown
    if npt_equilibration_parameter_file_name in parameter_files:
        npt_equilibration_parameter_file_name = npt_equilibration_parameter_file_name
    else:
        npt_equilibration_parameter_file_name = parameter_files[0] if parameter_files else None

    # Update production MD input file dropdown
    if npt_equilibration_run_input_file_name in run_input_files:
        prod_md_input_file_name = f"{npt_equilibration_run_input_file_name.split('.')[0]}.gro"
    else:
        prod_md_input_file_name = structure_files[0] if structure_files else None
    
    # Update production MD run input topology file dropdown
    if generate_ions_output_topology_file_name in topology_files:
        prod_md_input_topology_file_name = generate_ions_output_topology_file_name
    else:
        prod_md_input_topology_file_name = topology_files[0] if topology_files else None
    
    # Update production MD parameter file dropdown
    if prod_md_parameter_file_name in parameter_files:
        prod_md_parameter_file_name = prod_md_parameter_file_name
    else:
        prod_md_parameter_file_name = parameter_files[0] if parameter_files else None

    # Update production MD checkpoint file dropdown
    if prod_md_run_input_file_name in run_input_files:
        prod_md_checkpoint_file_name = f"{prod_md_run_input_file_name.split('.')[0]}.cpt"
    else:
        prod_md_checkpoint_file_name = checkpoint_files[0] if checkpoint_files else None

    # Update fix trajectory run input file dropdown
    if prod_md_run_input_file_name in run_input_files:
        fix_traj_run_input_file_name = prod_md_run_input_file_name
    else:
        fix_traj_run_input_file_name = run_input_files[0] if run_input_files else None

    # Update make molecule whole input trajectory file dropdown
    if prod_md_run_input_file_name in run_input_files:
        make_mol_whole_input_traj_file_name = f"{prod_md_run_input_file_name.split('.')[0]}.xtc"
    else:
        make_mol_whole_input_traj_file_name = trajectory_files[0] if trajectory_files else None

    # Update center protein input trajectory file dropdown
    if make_mol_whole_output_traj_file_name in trajectory_files:
        center_protein_input_traj_file_name = make_mol_whole_output_traj_file_name
    else:
        center_protein_input_traj_file_name = trajectory_files[0] if trajectory_files else None

    # Update fit backbone input trajectory file dropdown
    if center_protein_output_traj_file_name in trajectory_files:
        fit_backbone_input_traj_file_name = center_protein_output_traj_file_name
    else:
        fit_backbone_input_traj_file_name = trajectory_files[0] if trajectory_files else None

    # Update analysis input file dropdown
    if prod_md_run_input_file_name in run_input_files:
        analysis_structure_file_name = f"{prod_md_run_input_file_name.split('.')[0]}.gro"
    else:
        analysis_structure_file_name = structure_files[0] if structure_files else None

    # Update analysis input trajectory file dropdown
    if fit_backbone_output_traj_file_name in trajectory_files:
        analysis_input_traj_file_name = fit_backbone_output_traj_file_name
    else:
        analysis_input_traj_file_name = trajectory_files[0] if trajectory_files else None

    return file_df, \
        gr.update(choices=structure_files, value=protein_topology_input_file_name), \
        gr.update(choices=structure_files, value=ligand_topology_input_file_name), \
        gr.update(choices=structure_files, value=merge_structure_protein_input_file_name), \
        gr.update(choices=structure_files, value=merge_structure_ligand_input_file_name), \
        gr.update(choices=topology_files, value=merge_topology_protein_input_file_name), \
        gr.update(choices=topology_files, value=merge_topology_ligand_input_file_name), \
        gr.update(choices=structure_files, value=box_input_file_name), \
        gr.update(choices=structure_files, value=solvation_input_file_name), \
        gr.update(choices=topology_files, value=solvation_input_topology_file_name), \
        gr.update(choices=structure_files, value=generate_ions_input_file_name), \
        gr.update(choices=topology_files, value=generate_ions_input_topology_file_name), \
        gr.update(choices=parameter_files, value=generate_ions_parameter_file_name), \
        gr.update(choices=run_input_files, value=generate_ions_run_input_file_name), \
        gr.update(choices=structure_files, value=energy_minimization_input_file_name), \
        gr.update(choices=topology_files, value=energy_minimization_input_topology_file_name), \
        gr.update(choices=parameter_files, value=energy_minimization_parameter_file_name), \
        gr.update(choices=structure_files, value=nvt_equilibration_input_file_name), \
        gr.update(choices=topology_files, value=nvt_equilibration_input_topology_file_name), \
        gr.update(choices=parameter_files, value=nvt_equilibration_parameter_file_name), \
        gr.update(choices=structure_files, value=npt_equilibration_input_file_name), \
        gr.update(choices=topology_files, value=npt_equilibration_input_topology_file_name), \
        gr.update(choices=parameter_files, value=npt_equilibration_parameter_file_name), \
        gr.update(choices=structure_files, value=prod_md_input_file_name), \
        gr.update(choices=topology_files, value=prod_md_input_topology_file_name), \
        gr.update(choices=parameter_files, value=prod_md_parameter_file_name), \
        gr.update(choices=checkpoint_files, value=prod_md_checkpoint_file_name), \
        gr.update(choices=run_input_files, value=fix_traj_run_input_file_name), \
        gr.update(choices=trajectory_files, value=make_mol_whole_input_traj_file_name), \
        gr.update(choices=trajectory_files, value=center_protein_input_traj_file_name), \
        gr.update(choices=trajectory_files, value=fit_backbone_input_traj_file_name), \
        gr.update(choices=structure_files, value=analysis_structure_file_name), \
        gr.update(choices=trajectory_files, value=analysis_input_traj_file_name)

def on_select_file(evt: gr.SelectData):
    selected_file_name = evt.row_value[0]
    if selected_file_name.endswith('.pdb') or selected_file_name.endswith('.gro'):
        return selected_file_name, selected_file_name, None, gr.update(interactive=True)
    elif selected_file_name.endswith('.top') or selected_file_name.endswith('.itp') or selected_file_name.endswith('.mdp') or selected_file_name.endswith('.log'):
        return selected_file_name, None, selected_file_name, gr.update(interactive=True)
    else:
        return selected_file_name, None, None, gr.update(interactive=True)

def on_selected_structure_file_state_change(state):
    return gr.update(interactive=(state is not None))

def on_selected_text_file_state_change(state):
    return gr.update(interactive=(state is not None))

def on_delete_file(working_directory_path, selected_file_name):
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

def on_clean_working_directory(working_directory_path):
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

def on_view_protein_structure(working_directory_path, protein_file_name):
    try:
        protein_file_path = os.path.join(working_directory_path, protein_file_name)
        if protein_file_name.endswith('.gro'):
            protein_structure = pmd.load_file(protein_file_path)
            protein_structure.save("./static/complex_md_structure.pdb", overwrite=True)
            protein_file_path = "./static/complex_md_structure.pdb"
        
        # Create the NGL view widget
        view = nglview.show_structure_file(protein_file_path)
        view.clear()
        view.add_cartoon("protein", color="sstruc")
        view.add_ball_and_stick("LIG")
        view.add_ball_and_stick("NA")
        view.add_ball_and_stick("CL")
        view.add_ball_and_stick("SOL", opacity=0.3)

        # Write the widget to HTML
        if os.path.exists('./static/complex_md_structure.html'):
            os.remove('./static/complex_md_structure.html')
        nglview.write_html('./static/complex_md_structure.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/complex_md_structure.html?ts={timestamp}" height="800" width="600" title="NGL View"></iframe>'

        return html
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None

def on_view_text_file(working_directory_path, text_file_name):
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'r') as file:
            content = file.read()
        return gr.update(label=f"Text File Viewer - {text_file_name}", value=content, interactive=True), gr.update(interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None, None

def on_save_text_file(working_directory_path, text_file_name, text_content):
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

def on_upload_protein_structure_file(working_directory_path, protein_structure_file_name, protein_structure_file_path):
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

def on_upload_ligand_structure_file(working_directory_path, ligand_structure_file_name, ligand_structure_file_path):
    # Upload and rename the file
    save_file_path = os.path.join(working_directory_path, ligand_structure_file_name)
    try:
        if os.path.exists(save_file_path):
            os.remove(save_file_path)

        shutil.copy2(ligand_structure_file_path, save_file_path)

        status = "File uploaded successfully."
        return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"
    except Exception as exc:
        status = "Error uploading file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
def on_generate_protein_topology(working_directory_path, input_file_name, output_file_name, output_topology_file_name, force_field, water_model):
    try:
        cmd = [
            "gmx", "pdb2gmx",
            "-f", os.path.join(working_directory_path, input_file_name),
            "-o", os.path.join(working_directory_path, output_file_name),
            "-p", os.path.join(working_directory_path, output_topology_file_name),
            "-ff", force_field.lower(),
            "-water", water_model.lower(),
            "-ignh"
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "Topology generated successfully."
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ligand_topology(working_directory_path, ligand_input_file_name, ligand_output_file_name, ligand_charge, ligand_charge_model, ligand_force_field):
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

        subprocess.run(cmd, cwd=working_directory_path, check=True)
        
        # Copy ligand structure and topology files to working directory
        ligand_dir = os.path.join(working_directory_path, f'{ligand_output_file_name}.acpype')
        shutil.copy2(os.path.join(ligand_dir, f'{ligand_output_file_name}_GMX.gro'), os.path.join(working_directory_path, f'{ligand_output_file_name}_GMX.gro'))
        shutil.copy2(os.path.join(ligand_dir, f'{ligand_output_file_name}_GMX.itp'), os.path.join(working_directory_path, f'{ligand_output_file_name}_GMX.itp'))

        status = "Topology generated successfully."
    except Exception as exc:
        status = "Error generating topology!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_merge_structures(working_directory_path, protein_input_file, ligand_input_file, output_file):
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

def on_merge_topologies(working_directory_path, protein_input_file, ligand_input_file, output_file):
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

def on_merge_topology(working_directory_path, protein_input_file, ligand_input_file, output_file):
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

def on_generate_simulation_box(working_directory_path, input_file_name, output_file_name, box_type, distance):
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

        subprocess.run(cmd, check=True)
        status = "Simulation box generated successfully."
    except Exception as exc:
        status = "Error generating simulation box!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_solvate_protein(working_directory_path, input_file_name, output_file_name, input_topology_file_name, output_topology_file_name, solvent_configuration):
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

        subprocess.run(cmd, check=True)
        status = "Protein solvated successfully."
    except Exception as exc:
        status = "Error solvating protein!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ions_mdp_file(working_directory_path, parameter_file_name):
    file_content = get_default_ion_addition_mdp_file_content()
    file_path = os.path.join(working_directory_path, parameter_file_name)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Ion addition parameter file generated successfully."
    except Exception as exc:
        status = "Error generating ion addition parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_ions_tpr_file(working_directory_path, input_file_name, input_topology_file_name, parameter_file_name, run_input_file_name):
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", "5"
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "Ion addition run input file generated successfully."
    except Exception as exc:
        status = "Error generating ion addition run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_add_ions_method_change(add_ions_method):
    if add_ions_method == "Concentration":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:  # add_ions_method == "Number"
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def on_add_ions(working_directory_path, run_input_file_name, output_file_name, input_topology_file_name, output_topology_file_name, cation_name, anion_name, add_ion_method, concentration, cation_charge, anion_charge, number_of_cations, number_of_anions, neutralize):
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

        # genion requires user input to select a group; we will provide "SOL" (usually group 13)
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate(input="15\n")

        if process.returncode != 0:
            raise Exception(stderr)

        status = "Ions added successfully."
    except Exception as exc:
        status = "Error adding ions!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_energy_minimization_mdp_file(working_directory_path, parameter_file_name):
    file_content = get_default_energy_minimization_mdp_file_content()
    file_path = os.path.join(working_directory_path, parameter_file_name)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Energy minimization parameter file generated successfully."
    except Exception as exc:
        status = "Error generating energy minimization parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_energy_minimization_tpr_file(working_directory_path, input_file_name, input_topology_file_name, parameter_file_name, run_input_file_name):
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", "5"
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "Energy minimization run input file generated successfully."
    except Exception as exc:
        status = "Error generating energy minimization run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_energy_minimization(working_directory_path, run_input_file_name, mpi_rank, omp_threads):
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", os.path.join(working_directory_path, base_name),
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ]

        subprocess.run(cmd, check=True)
        status = "Energy minimization completed successfully."
    except Exception as exc:
        status = "Error during energy minimization!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_mdp_file(working_directory_path, time_scale, time_step, temperature, parameter_file_name):
    file_content = get_default_nvt_equilibration_mdp_file_content(time_scale_ps=time_scale, time_step_ps=time_step, temperature=temperature, with_ligand=True)
    file_path = os.path.join(working_directory_path, parameter_file_name)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "NVT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NVT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_nvt_equilibration_tpr_file(working_directory_path, input_file_name, input_topology_file_name, parameter_file_name, run_input_file_name):
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", "5"
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "NVT equilibration run input file generated successfully."
    except Exception as exc:
        status = "Error generating NVT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def watch_process(proc, process_state):
    proc.wait()  # wait until finished

    # If user already stopped it, do nothing
    if not process_state["running"]:
        return

    # Process finished naturally
    process_state["proc"] = None
    process_state["running"] = False

def sync_button_state(process_state):
    if process_state["running"]:
        return gr.update(value="Stop", variant="stop")
    else:
        return gr.update(value="Start", variant="primary")
    
def on_run_nvt_equilibration(working_directory_path, run_input_file_name, mpi_rank, omp_threads, process_state):
    # ---------- STOP ----------
    if process_state["running"]:
        proc = process_state["proc"]
        if proc and proc.poll() is None:
            proc.kill()

        process_state["proc"] = None
        process_state["running"] = False

        status = "NVT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", os.path.join(working_directory_path, base_name),
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ]

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd='.', text=True)

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
        process_state["proc"] = None
        process_state["running"] = False

        status = f"Error during NVT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_generate_npt_equilibration_mdp_file(working_directory_path, time_scale, time_step, temperature, pressure, parameter_file_name):
    file_content = get_default_npt_equilibration_mdp_file_content(time_scale_ps=time_scale, time_step_ps=time_step, temperature=temperature, pressure=pressure, with_ligand=True)
    file_path = os.path.join(working_directory_path, parameter_file_name)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "NPT equilibration parameter file generated successfully."
    except Exception as exc:
        status = "Error generating NPT equilibration parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_npt_equilibration_tpr_file(working_directory_path, input_file_name, input_topology_file_name, parameter_file_name, run_input_file_name):
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-r", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", "5"
        ]
        
        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "NPT equilibration run input file generated successfully."
    except Exception as exc:
        status = "Error generating NPT equilibration run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_npt_equilibration(working_directory_path, run_input_file_name, mpi_rank, omp_threads, process_state):
    # ---------- STOP ----------
    if process_state["running"]:
        proc = process_state["proc"]
        if proc and proc.poll() is None:
            proc.kill()

        process_state["proc"] = None
        process_state["running"] = False

        status = "NPT equilibration stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", os.path.join(working_directory_path, base_name),
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ]

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd='.', text=True)

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
        process_state["proc"] = None
        process_state["running"] = False

        status = f"Error during NPT equilibration:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_change_mdp_type(prod_md_mdp_type_radio):
    if prod_md_mdp_type_radio=="Initial":
        return gr.update(visible=True), "md_initial.mdp"
    else:
        return gr.update(visible=False), "md_continue.mdp"

def on_generate_prod_md_mdp_file(working_directory_path, time_scale, time_step, temperature, pressure, mdp_type, random_seed, pparameter_file_name):
    file_content = get_default_prod_md_mdp_file_content(time_scale_ps=time_scale*1000, time_step_ps=time_step, temperature=temperature, pressure=pressure, mdp_type=mdp_type, random_seed=random_seed, with_ligand=True)
    file_path = os.path.join(working_directory_path, pparameter_file_name)
    try:
        with open(file_path, 'w') as file:
            file.write(file_content)
        status = "Production MD parameter file generated successfully."
    except Exception as exc:
        status = "Error generating production MD parameter file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_generate_prod_md_tpr_file(working_directory_path, input_file_name, input_topology_file_name, parameter_file_name, run_input_file_name):
    try:
        cmd = [
            "gmx", "grompp",
            "-f", os.path.join(working_directory_path, parameter_file_name),
            "-c", os.path.join(working_directory_path, input_file_name),
            "-p", os.path.join(working_directory_path, input_topology_file_name),
            "-o", os.path.join(working_directory_path, run_input_file_name),
            "-maxwarn", "5"
        ]

        print(f"Running command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        status = "Production MD run input file generated successfully."
    except Exception as exc:
        status = "Error generating production MD run input file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_run_prod_md(working_directory_path, run_input_file_name, mpi_rank, omp_threads, use_gpu, process_state):
    # ---------- STOP ----------
    if process_state["running"]:
        proc = process_state["proc"]
        if proc and proc.poll() is None:
            proc.kill()

        process_state["proc"] = None
        process_state["running"] = False

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", os.path.join(working_directory_path, base_name),
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-v"
        ]
        if use_gpu:
            cmd.extend([
                "-nb", "gpu",
                "-pme", "gpu",
                "-bonded", "gpu",
                "-update", "gpu",
                "-pin", "on",
                "-dlb", "yes"
            ])

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd='.', text=True)

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
        process_state["proc"] = None
        process_state["running"] = False

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

def on_continue_prod_md(working_directory_path, run_input_file_name, checkpoint_file_name, mpi_rank, omp_threads, use_gpu, process_state):
    # ---------- STOP ----------
    if process_state["running"]:
        proc = process_state["proc"]
        if proc and proc.poll() is None:
            proc.kill()

        process_state["proc"] = None
        process_state["running"] = False

        status = "Production MD stopped by user."

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")

    # ---------- START ----------
    try:
        base_name = os.path.splitext(run_input_file_name)[0]

        cmd = [
            "gmx", "mdrun",
            "-deffnm", os.path.join(working_directory_path, base_name),
            "-cpi", os.path.join(working_directory_path, checkpoint_file_name),
            "-ntmpi", str(mpi_rank),
            "-ntomp", str(omp_threads),
            "-append",
            "-v"
        ]
        if use_gpu:
            cmd.extend([
                "-nb", "gpu",
                "-pme", "gpu",
                "-bonded", "gpu",
                "-update", "gpu",
                "-pin", "on",
                "-dlb", "yes"
            ])

        print(f"Running command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, cwd='.', text=True)

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
        process_state["proc"] = None
        process_state["running"] = False

        status = f"Error during Production MD:<br>{exc}"

        return get_files_in_working_directory(working_directory_path), f"<span style='color:red;'>{status}</span>", process_state, gr.update(value="Start", variant="primary")
    
def on_make_molecule_whole(working_directory_path, run_input_file_name, input_traj_file_name, output_traj_file_name):
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
        subprocess.run(cmd, input="0\n", text=True, check=True)
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_center_protein(working_directory_path, run_input_file_name, input_traj_file_name, output_traj_file_name):
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
        subprocess.run(cmd, input="1\n0\n", text=True, check=True)
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_fit_backbone(working_directory_path, run_input_file_name, input_traj_file_name, output_traj_file_name):
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
        subprocess.run(cmd, input="4\n0\n", text=True, check=True)
        
        status = "Operation executed successfully."
    except Exception as exc:
        status = "Error fixing trajectory!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"
        
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def on_analyze_md_traj(working_directory_path, structure_file_name, input_traj_file_name):
    u = mda.Universe(os.path.join(working_directory_path, structure_file_name), os.path.join(working_directory_path, input_traj_file_name))
    protein_selector = u.select_atoms("protein")
    ligand_selector = u.select_atoms("resname LIG")
    
    # Calculate protein RMSD
    protein_rmsd = rms.RMSD(
        u,
        select="protein and backbone",
        groupselections=["protein"],
        ref_frame=0
    ).run()
    protein_rmsd_values = protein_rmsd.results.rmsd[:,2]

    # Calculate ligand RMSD
    ligand_rmsd = rms.RMSD(
        u,
        select="resname LIG",
        groupselections=["resname LIG"],
        ref_frame=0,
        rmsd_kwargs={"center": True, "superposition": True}
    ).run()
    ligand_rmsd_values = ligand_rmsd.results.rmsd[:,2]

    time_ns = protein_rmsd.results.rmsd[:,1] / 1000
    rmsd_df = pd.DataFrame({"Time (ns)": time_ns, "Protein RMSD (Å)": protein_rmsd_values, "Ligand RMSD (Å)": ligand_rmsd_values})
    
    # Plot RMSD
    rmsd_fig = plt.figure(figsize=(8, 6))
    plt.plot(time_ns, protein_rmsd_values, label="Protein")
    plt.plot(time_ns, ligand_rmsd_values, label="Ligand")
    plt.xlabel("Time (ns)")
    plt.ylabel("RMSD (Å)")
    plt.legend()
    plt.title("RMSD vs Time")
    plt.tight_layout()
    
    # Calculate protein–ligand center of mass distance
    com_dist = []
    for _ in u.trajectory:
        d = np.linalg.norm(
            protein_selector.center_of_mass() - ligand_selector.center_of_mass()
        )
        com_dist.append(d)
    
    com_dist_df = pd.DataFrame({"Time (ns)": time_ns, "Center of mass distance (Å)": com_dist})
    
    com_dist_fig = plt.figure(figsize=(8, 6))
    plt.plot(time_ns, com_dist)
    plt.xlabel("Time (ns)")
    plt.ylabel("Center of mass distance (Å)")

    # Calculate mean Cα RMSF
    ca_selector = u.select_atoms("protein and name CA")
    RMSF_ca = rms.RMSF(ca_selector).run()
    ca_rmsf = RMSF_ca.results.rmsf

    mean_ca_rmsf = ca_rmsf.mean()
    cd_rmsf_df = pd.DataFrame({"Residue Index": ca_selector.resids, "Cα RMSF (Å)": ca_rmsf})

    ca_rmsf_fig = plt.figure(figsize=(8, 6))
    plt.plot(ca_selector.residues.resids, ca_rmsf, label="Cα RMSF")
    plt.axhline(mean_ca_rmsf, color="red", linestyle="--", label="Mean Cα RMSF")

    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.title("Cα RMSF per Residue")
    plt.legend()
    plt.tight_layout()

    return html, rmsd_df, rmsd_fig, com_dist_df, com_dist_fig, cd_rmsf_df, ca_rmsf_fig

def on_export_df(working_directory_path, df, file_name):
    try:
        df.to_csv(os.path.join(working_directory_path, file_name), index=False)
        status = f"File exported: {file_name}"
    except Exception as exc:
        status = "Error exporting file!\n" + str(exc)
        return get_files_in_working_directory(working_directory_path), "<span style='color:red;'>" + status + "</span>"  
    
    return get_files_in_working_directory(working_directory_path), "<span style='color:green;'>" + status + "</span>"

def protein_ligand_complex_md_simulation_tab_content():
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
                view_structure_button = gr.Button(value="View Structure", interactive=False)
                structure_viewer_html = gr.HTML()
                view_text_file_button = gr.Button(value="View Text File", interactive=False)
                text_file_viewer_textarea = gr.TextArea(label="Text File Viewer", lines=20, elem_id="textfile_viewer", interactive=False)
                save_text_file_button = gr.Button(value="Save Text File", interactive=False)
            with gr.Column(scale=2):
                with gr.Row():
                    status_markdown = gr.Markdown()
                with gr.Accordion(label="Settings", open=False):
                    with gr.Row():
                        use_gpu = gr.Checkbox(label="Use GPU", value="True")
                        mpi_rank_slider = gr.Slider(label="MPI Ranks", minimum=1, maximum=psutil.cpu_count(logical=False), value=1, step=1)
                        omp_threads_slider = gr.Slider(label="OpenMP Threads", minimum=1, maximum=int(os.environ.get("OMP_NUM_THREADS")), value=int(os.environ.get("OMP_NUM_THREADS")), step=1)
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
                                                                                                    "CHARMM27", "GROMOS43A1", "GROMOS43A2", "GROMOS45A3", "GROMOS53A5", "GROMOS53A6", "GROMOS54A7", ("OPLS-AA", "OPLSAA")], value="AMBER99SB-ILDN", allow_custom_value=True)
                                    water_model_dropdown = gr.Dropdown(label="Water Model", choices=["SELECT", "NONE", "OPC", "OPC3", "SPC", "SPCE", "TIP3P", "TIP4P", ("TIP4P-Ew", "TIP4PEW"), "TIP5P", "TIPS3P"], value="TIP3P")
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
                                nvt_process_state = gr.State({"proc": None, "running": False})
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
                                npt_process_state = gr.State({"proc": None, "running": False})
                                run_npt_equilibration_button = gr.Button(value="Run NPT Equilibration")
                                npt_equilibration_timer = gr.Timer(1.0)
                with gr.Accordion(label="Production MD", open=False):
                    with gr.Row():
                        with gr.Column():    
                            with gr.Row():
                                gr.Markdown("***Generate parameter file for production MD simulation***")
                            with gr.Row():
                                with gr.Column():
                                    prod_md_time_scale_slider = gr.Slider(label="Production MD Time (ns)", minimum=1, maximum=1000, value=500, step=1)
                                    prod_md_time_step_slider = gr.Slider(label="Time Step (ps)", minimum=0.001, maximum=0.005, value=0.002, step=0.001)
                                    prod_md_temperature_slider = gr.Slider(label="Target Temperature (K)", minimum=100, maximum=500, value=300, step=10)
                                    prod_md_pressure_slider = gr.Slider(label="Pressure (atm)", minimum=0.1, maximum=10, value=1, step=0.1)
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
                                    gr.Markdown("*Run from beginning*")
                                    prod_md_initial_process_state = gr.State({"proc": None, "running": False})
                                    run_prod_md_button = gr.Button(value="Run production MD simulation")
                                    prod_md_initial_timer = gr.Timer(1.0)
                                with gr.Column():
                                    gr.Markdown("*Run from a checkpoint*")
                                    prod_md_continuation_process_state = gr.State({"proc": None, "running": False})
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
                    with gr.Row():
                        analysis_structure_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                        analysis_input_traj_file_name_dropdown = gr.Dropdown(label="Input Trajectory File Name", choices=[], value=None)
                        analyze_button = gr.Button("Analyze")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("***Protein RMSD***")
                            rmsd_df_state = gr.State()
                            rmsd_plot = gr.Plot()
                            with gr.Row():
                                rmsd_file_name_texbox = gr.Textbox(label="RMSD File Name", value="RMSD.csv")
                                rmsd_export_button = gr.Button("Export RMSD (.csv)")
                        with gr.Column():
                            gr.Markdown("***Center of mass distance***")
                            com_dist_df_state = gr.State()
                            com_dist_plot = gr.Plot()
                            with gr.Row():
                                com_dist_file_name_texbox = gr.Textbox(label="COM Distance File Name", value="COM_distance.csv")
                                com_dist_export_button = gr.Button("Export COM Distance (.csv)")
                        with gr.Column():
                            gr.Markdown("***Cα RMSF***")
                            ca_rmsf_df_state = gr.State()
                            ca_rmsf_plot = gr.Plot()
                            with gr.Row():
                                ca_rmsf_file_name_texbox = gr.Textbox(label="Cα RMSF File Name", value="C_alpha_RMSF.csv")
                                ca_rmsf_export_button = gr.Button("Export Cα RMSF (.csv)")

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
                                             energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown,
                                             nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown,
                                             npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown,
                                             prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, checkpoint_file_dropdown,
                                             fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, center_protein_input_traj_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown,
                                             analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown])
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, view_structure_button)
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, view_text_file_button)
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    clean_working_directory_button.click(on_clean_working_directory, working_directory_path_state, working_directory_file_list_state)
    view_structure_button.click(on_view_protein_structure, [working_directory_path_state, selected_structure_file_state], structure_viewer_html)
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, selected_text_file_state, text_file_viewer_textarea], working_directory_file_list_state)

    # Protein and ligand structure file upload interaction
    protein_structure_file.upload(on_upload_protein_structure_file, [working_directory_path_state, protein_structure_file_name_textbox, protein_structure_file], [working_directory_file_list_state, status_markdown])
    ligand_structure_file.upload(on_upload_ligand_structure_file, [working_directory_path_state, ligand_structure_file_name_textbox, ligand_structure_file], [working_directory_file_list_state, status_markdown])
    
    # Generate protein and ligand topology interaction
    generate_protein_topology_button.click(on_generate_protein_topology, [working_directory_path_state, protein_topology_input_file_name_dropdown, protein_topology_output_file_name_textbox, protein_topology_output_topology_file_name_textbox, protein_force_field_dropdown, water_model_dropdown], [working_directory_file_list_state, status_markdown])
    generate_ligand_topology_button.click(on_generate_ligand_topology, [working_directory_path_state, ligand_topology_input_file_name_dropdown, ligand_output_file_name_textbox, ligand_charge_slider, ligand_charge_model_dropdown, ligand_force_field_dropdown], [working_directory_file_list_state, status_markdown])

    # Merge structure and topology interaction
    merge_structures_button.click(on_merge_structures, [working_directory_path_state, merge_structures_protein_input_file_name_dropdown, merge_structures_ligand_input_file_name_dropdown, merge_structures_output_file_name_textbox], [working_directory_file_list_state, status_markdown])
    merge_topologies_button.click(on_merge_topologies, [working_directory_path_state, merge_topologies_protein_input_file_name_dropdown, merge_topologies_ligand_input_file_name_dropdown, merge_topologies_output_file_name_textbox], [working_directory_file_list_state, status_markdown])

    # Generate simulation box interaction
    generate_box_button.click(on_generate_simulation_box, [working_directory_path_state, box_input_file_name_dropdown, box_output_file_name_textbox, box_type_dropdown, distance_slider], [working_directory_file_list_state, status_markdown])

    # Solvation interaction
    solvate_button.click(on_solvate_protein, [working_directory_path_state, solvation_input_file_name_dropdown, solvation_output_file_name_textbox, solvation_input_topology_file_name_dropdown, solvation_output_topology_file_name_textbox, solvent_configuration_dropdown], [working_directory_file_list_state, status_markdown])

    # Generate ions interaction
    generate_ions_parameter_file_button.click(on_generate_ions_mdp_file, [working_directory_path_state, generate_ions_parameter_file_name_textbox], [working_directory_file_list_state, status_markdown])
    generate_ions_run_input_file_button.click(on_generate_ions_tpr_file, [working_directory_path_state, generate_ions_input_file_name_dropdown, generate_ions_input_topology_file_name_dropdown, generate_ions_parameter_file_dropdown, generate_ions_run_input_file_name_textbox], [working_directory_file_list_state, status_markdown])
    add_ion_method_radio.change(on_add_ions_method_change, add_ion_method_radio, [concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider])
    add_ions_button.click(on_add_ions, [working_directory_path_state, generate_ions_run_input_file_dropdown, generate_ions_output_file_name_textbox, generate_ions_input_topology_file_name_dropdown, generate_ions_output_topology_file_name_textbox, cation_name_textbox, anion_name_textbox, add_ion_method_radio, concentration_slider, cation_charge_slider, anion_charge_slider, number_of_cations_slider, number_of_anions_slider, netralize_checkbox], [working_directory_file_list_state, status_markdown])
    
    # Energy minimization interaction
    energy_minimization_parameter_file_button.click(on_generate_energy_minimization_mdp_file, [working_directory_path_state, energy_minimization_parameter_file_name_textbox], [working_directory_file_list_state, status_markdown])
    energy_minimization_run_input_file_button.click(on_generate_energy_minimization_tpr_file, [working_directory_path_state, energy_minimization_input_file_name_dropdown, energy_minimization_input_topology_file_name_dropdown, energy_minimization_parameter_file_dropdown, energy_minimization_run_input_file_name_textbox], [working_directory_file_list_state, status_markdown])
    run_energy_minimization_button.click(on_run_energy_minimization, [working_directory_path_state, energy_minimization_run_input_file_name_textbox, mpi_rank_slider, omp_threads_slider], [working_directory_file_list_state, status_markdown])

    # NVT equilibration interaction
    nvt_equilibration_parameter_file_button.click(on_generate_nvt_equilibration_mdp_file, [working_directory_path_state, nvt_time_scale_slider, nvt_time_step_slider, nvt_temperature_slider, nvt_equilibration_parameter_file_name_textbox], [working_directory_file_list_state, status_markdown])
    nvt_equilibration_run_input_file_button.click(on_generate_nvt_equilibration_tpr_file, [working_directory_path_state, nvt_equilibration_input_file_name_dropdown, nvt_equilibration_input_topology_file_name_dropdown, nvt_equilibration_parameter_file_dropdown, nvt_equilibration_run_input_file_name_textbox], [working_directory_file_list_state, status_markdown])
    run_nvt_equilibration_button.click(on_run_nvt_equilibration, [working_directory_path_state, nvt_equilibration_run_input_file_name_textbox, mpi_rank_slider, omp_threads_slider, nvt_process_state], [working_directory_file_list_state, status_markdown, nvt_process_state, run_nvt_equilibration_button])
    nvt_equilibration_timer.tick(sync_button_state, nvt_process_state, run_nvt_equilibration_button)

    # NPT equilibration interaction
    npt_equilibration_parameter_file_button.click(on_generate_npt_equilibration_mdp_file, [working_directory_path_state, npt_time_scale_slider, npt_time_step_slider, npt_temperature_slider, npt_pressure_slider, npt_equilibration_parameter_file_name_textbox], [working_directory_file_list_state, status_markdown])
    npt_equilibration_run_input_file_button.click(on_generate_npt_equilibration_tpr_file, [working_directory_path_state, npt_equilibration_input_file_name_dropdown, npt_equilibration_input_topology_file_name_dropdown, npt_equilibration_parameter_file_dropdown, npt_equilibration_run_input_file_name_textbox], [working_directory_file_list_state, status_markdown])
    run_npt_equilibration_button.click(on_run_npt_equilibration, [working_directory_path_state, npt_equilibration_run_input_file_name_textbox, mpi_rank_slider, omp_threads_slider, npt_process_state], [working_directory_file_list_state, status_markdown, npt_process_state, run_npt_equilibration_button])
    npt_equilibration_timer.tick(sync_button_state, npt_process_state, run_npt_equilibration_button)

    # Production MD interaction
    prod_md_mdp_type_radio.change(on_change_mdp_type, prod_md_mdp_type_radio, [prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox])
    prod_md_parameter_file_button.click(on_generate_prod_md_mdp_file, [working_directory_path_state, prod_md_time_scale_slider, prod_md_time_step_slider, prod_md_temperature_slider, prod_md_pressure_slider, prod_md_mdp_type_radio, prod_md_random_seed_textbox, prod_md_parameter_file_name_textbox], [working_directory_file_list_state, status_markdown])
    prod_md_run_input_file_button.click(on_generate_prod_md_tpr_file, [working_directory_path_state, prod_md_input_file_name_dropdown, prod_md_input_topology_file_name_dropdown, prod_md_parameter_file_dropdown, prod_md_run_input_file_name_textbox], [working_directory_file_list_state, status_markdown])
    run_prod_md_button.click(on_run_prod_md, [working_directory_path_state, prod_md_run_input_file_name_textbox, mpi_rank_slider, omp_threads_slider, use_gpu, prod_md_initial_process_state], [working_directory_file_list_state, status_markdown, prod_md_initial_process_state, run_prod_md_button])
    prod_md_initial_timer.tick(sync_button_state, prod_md_initial_process_state, run_prod_md_button)
    continue_prod_md_button.click(on_continue_prod_md, [working_directory_path_state, prod_md_run_input_file_name_textbox, checkpoint_file_dropdown, mpi_rank_slider, omp_threads_slider, use_gpu, prod_md_continuation_process_state], [working_directory_file_list_state, status_markdown, prod_md_continuation_process_state, continue_prod_md_button])
    prod_md_continuation_timer.tick(sync_button_state, prod_md_continuation_process_state, continue_prod_md_button)

    # Fix trajectory interaction
    make_mol_whole_button.click(on_make_molecule_whole, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, make_mol_whole_input_traj_file_name_dropdown, make_mol_whole_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    center_protein_button.click(on_center_protein, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, center_protein_input_traj_file_name_dropdown, center_protein_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])
    fit_backbone_button.click(on_fit_backbone, [working_directory_path_state, fix_traj_run_input_file_name_dropdown, fit_backbone_input_traj_file_name_dropdown, fit_backbone_output_traj_file_name_textbox], [working_directory_file_list_state, status_markdown])

    # Analysis
    analyze_button.click(on_analyze_md_traj, [working_directory_path_state, analysis_structure_file_name_dropdown, analysis_input_traj_file_name_dropdown], [rmsd_df_state, rmsd_plot, com_dist_df_state, com_dist_plot, ca_rmsf_df_state, ca_rmsf_plot])
    rmsd_export_button.click(on_export_df, [working_directory_path_state, rmsd_df_state, rmsd_file_name_texbox], [working_directory_file_list_state, status_markdown])
    ca_rmsf_export_button.click(on_export_df, [working_directory_path_state, ca_rmsf_df_state, ca_rmsf_file_name_texbox], [working_directory_file_list_state, status_markdown])
    com_dist_export_button.click(on_export_df, [working_directory_path_state, com_dist_df_state, com_dist_file_name_texbox], [working_directory_file_list_state, status_markdown])
    
    return protein_ligand_complex_md_simulation_tab