import os
import threading

class ProcessStateDict(dict):
    """dict subclass for gr.State that creates a fresh lock on deep copy."""
    def __init__(self):
        super().__init__({"proc": None, "running": False, "lock": threading.Lock()})

    def __deepcopy__(self, memo):
        return ProcessStateDict()

def get_default_ion_addition_mdp_file_content():
    return """
integrator = steep
nsteps     = 500
emtol      = 1000.0

cutoff-scheme = Verlet
coulombtype   = PME
rcoulomb      = 1.0
rvdw          = 1.0
"""

def get_default_energy_minimization_mdp_file_content():
    return """
integrator  = steep
nsteps      = 50000
emtol       = 1000.0
"""

def get_default_nvt_equilibration_mdp_file_content(time_scale_ps=500, time_step_ps=0.002, temperature=300, with_ligand=False):
    if with_ligand:
        tc_grps_value = "Protein LIG Water_and_ions"
        tau_t_value = "0.1 0.1 0.1"
        temperature_value = f"{temperature} {temperature} {temperature}"
    else:
        tc_grps_value = "Protein Water_and_ions"
        tau_t_value = "0.1 0.1"
        temperature_value = f"{temperature} {temperature}"

    return f"""
integrator  = md
dt          = {time_step_ps}
nsteps      = {int(time_scale_ps / time_step_ps)}
tcoupl      = V-rescale
tc-grps     = {tc_grps_value}
tau_t       = {tau_t_value}
ref_t       = {temperature_value}
constraints = h-bonds
"""

def get_default_npt_equilibration_mdp_file_content(time_scale_ps=1000, time_step_ps=0.002, temperature=300, pressure=1.0, with_ligand=False):
    if with_ligand:
        tc_grps_value = "Protein LIG Water_and_ions"
        tau_t_value = "0.1 0.1 0.1"
        temperature_value = f"{temperature} {temperature} {temperature}"
    else:
        tc_grps_value = "Protein Water_and_ions"
        tau_t_value = "0.1 0.1"
        temperature_value = f"{temperature} {temperature}"
    
    return f"""
integrator      = md
dt              = {time_step_ps}
nsteps          = {int(time_scale_ps / time_step_ps)}

; Output control
nstxout         = 1000
nstvout         = 1000
nstenergy       = 1000
nstlog          = 1000

; Temperature coupling
tcoupl          = V-rescale
tc-grps         = {tc_grps_value}
tau_t           = {tau_t_value}
ref_t           = {temperature_value}

; Pressure coupling
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {pressure}
compressibility = 4.5e-5

; Constraints
constraints     = h-bonds
constraint_algorithm = lincs

; Cutoffs
cutoff-scheme   = Verlet
rlist           = 1.0
rvdw            = 1.0
rcoulomb        = 1.0
coulombtype     = PME
"""

def get_default_prod_md_mdp_file_content(time_scale_ps=1000, time_step_ps=0.002, temperature=300, pressure=1.0, mdp_type="Initial", random_seed=0, with_ligand=False):
    if with_ligand:
        tc_grps_value = "Protein LIG Water_and_ions"
        tau_t_value = "0.1 0.1 0.1"
        temperature_value = f"{temperature} {temperature} {temperature}"
    else:
        tc_grps_value = "Protein Water_and_ions"
        tau_t_value = "0.1 0.1"
        temperature_value = f"{temperature} {temperature}"
    
    content = f"""
integrator      = md
dt              = {time_step_ps}
nsteps          = {int(time_scale_ps / time_step_ps)}

; Output
nstxout         = 5000
nstvout         = 5000
nstenergy       = 5000
nstlog          = 5000
nstxout-compressed = 5000

; Neighbor searching
cutoff-scheme   = Verlet
rlist           = 1.0
rvdw            = 1.0
rcoulomb        = 1.0
coulombtype     = PME

; Temperature coupling
tcoupl          = V-rescale
tc-grps         = {tc_grps_value}
tau_t           = {tau_t_value}
ref_t           = {temperature_value}

; Pressure coupling
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {pressure}
compressibility = 4.5e-5

; Constraints
constraints     = h-bonds
constraint_algorithm = lincs
"""
    if mdp_type=="Initial":
        content = content + f"""
; Continuation
continuation    = no
gen_vel         = yes
gen_temp        = {temperature}
gen_seed        = {random_seed}
"""
    else: # mdp_type=="Continuation"
        content = content + f"""
; Continuation
continuation    = yes
"""

    return content

def read_gromacs_structure_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    title = lines[0].strip()
    natoms = int(lines[1].strip())
    atoms = lines[2:2 + natoms]
    box = lines[2 + natoms].strip()

    return title, natoms, atoms, box

def merge_protein_ligand_structures(protein_structure_file_path, ligand_structure_file_path, output_structure_file_path):
    # Read input files
    _, p_n, p_atoms, p_box = read_gromacs_structure_file(protein_structure_file_path)
    _, l_n, l_atoms, _ = read_gromacs_structure_file(ligand_structure_file_path)

    # Combination
    total_atoms = p_n + l_n

    with open(output_structure_file_path, "w") as out:
        out.write("Protein + ligand complex\n")
        out.write(f"{total_atoms}\n")

        for line in p_atoms:
            out.write(line)

        for line in l_atoms:
            out.write(line)

        # keep protein box (ligand box is meaningless alone)
        out.write(p_box + "\n")

def merge_protein_ligand_topologies(protein_topology_file_path, ligand_topology_file_path, output_topology_file_path):
    ligand_topology_file_name = os.path.basename(ligand_topology_file_path)

    with open(protein_topology_file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    ligand_include_added = False
    molecules_section_found = False
    ligand_in_molecules = False

    for _, line in enumerate(lines):
        new_lines.append(line)

        # 1Insert ligand include after forcefield.itp
        if (
            not ligand_include_added
            and line.strip().startswith('#include')
            and 'forcefield.itp' in line
        ):
            new_lines.append('\n; Include ligand topology\n')
            new_lines.append(f'#include "{ligand_topology_file_name}"\n')
            ligand_include_added = True

        # Detect [ molecules ] section
        if line.strip().lower() == "[ molecules ]":
            molecules_section_found = True

        # Check if ligand already listed
        if molecules_section_found:
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == "ligand":
                ligand_in_molecules = True

    # 2️Append ligand to [ molecules ]
    if not molecules_section_found:
        new_lines.append('\n[ molecules ]\n')
        new_lines.append('; Compound        #mols\n')

    if not ligand_in_molecules:
        new_lines.append('ligand            1\n')

    # Write merged topology
    with open(output_topology_file_path, "w") as f:
        f.writelines(new_lines)       