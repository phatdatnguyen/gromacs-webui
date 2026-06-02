import os
import threading
import torch
from nnpot_models import (
    GmxAIMNet2Model,
    GmxANI1xModel,
    GmxANI2xEMLEModel,
    GmxANI2xModel,
    GmxMACEModel,
)
from e3nn.util.jit import script

def get_torchani_install_error_message(exc):
    message = str(exc)
    if "PERIODIC_TABLE" not in message or "torchani.utils" not in message:
        return None

    return (
        "TorchANI could not be imported because the current environment appears "
        "to contain a mixed or stale TorchANI installation. This usually happens "
        "after installing the newer pip package over the older conda package, "
        "leaving incompatible files from both versions in site-packages.\n\n"
        "Repair the conda environment, then try generating the MDP again:\n"
        "  python -m pip uninstall -y torchani\n"
        "  python -m pip uninstall -y torchani\n"
        "  python -m pip install torchani\n"
        "  ani build-extensions\n\n"
        f"Original import error: {message}"
    )

def get_emle_install_error_message(model_name, exc):
    if model_name != "ani2x-emle":
        return None

    message = str(exc)
    if "pygit2" in message:
        return (
            "EMLE could not fetch its model resources because the Python "
            "package 'pygit2' is missing.\n\n"
            "Install it in the active environment, then regenerate the MDP:\n"
            "  python -m pip install pygit2\n\n"
            f"Original import error: {message}"
        )
    if "SpeciesEnergies" in message and "torchani" in message:
        return (
            "EMLE could not import TorchANI compatibility classes. The wrapper "
            "adds a TorchANI 2.8 compatibility shim, but this EMLE/TorchANI "
            "combination still appears incompatible.\n\n"
            "Try reinstalling EMLE after TorchANI, or use the TorchANI version "
            "recommended by your EMLE checkout.\n\n"
            f"Original import error: {message}"
        )

    return None

def get_nnpot_model_load_error_message(exc):
    message = str(exc)
    if "torch.classes.cuaev.CuaevComputer" not in message:
        return None

    return (
        "The cached ANI model was scripted with TorchANI's optional cuAEV "
        "extension, but that custom class is not available when the model is "
        "loaded. Rebuild the cached ANI model with the pure PyTorch AEV "
        "strategy.\n\n"
        f"Original model load error: {message}"
    )

def get_expected_nnpot_model_config(model_name):
    if model_name in ["ani1x", "ani2x"]:
        return f"{model_name}|torchani|pyaev|adaptive|extensions-disabled"
    if model_name == "ani2x-emle":
        return f"{model_name}|emle|empty-mm-environment|energy-only-pyaev-v2"
    if model_name.startswith("mace-"):
        return f"{model_name}|mace|internal-neighbors-singular-cell-v4"
    if model_name == "aimnet2":
        return f"{model_name}|aimnet|traced-positions-numbers-box-pbc-device-float64-v5"
    return model_name

def is_cached_nnpot_model_usable(model_name, modelfile_path):
    extra_files = {"nnpot_model_config": ""}
    try:
        torch.jit.load(modelfile_path, map_location="cpu", _extra_files=extra_files)
        cached_config = extra_files["nnpot_model_config"]
        if isinstance(cached_config, bytes):
            cached_config = cached_config.decode()
        if cached_config != get_expected_nnpot_model_config(model_name):
            backup_path = modelfile_path + ".invalid"
            os.replace(modelfile_path, backup_path)
            print(f"Moved outdated cached NNPot model to {backup_path}.")
            return False
        return True
    except RuntimeError as exc:
        if get_nnpot_model_load_error_message(exc) is not None:
            backup_path = modelfile_path + ".invalid"
            os.replace(modelfile_path, backup_path)
            print(f"Moved unusable cached NNPot model to {backup_path}.")
            return False
        raise

def checkExtensions():
    ext_lib = []
    for lib in torch.ops.loaded_libraries:
        if lib:
            ext_lib.append(lib)
    ext_lib = ":".join(ext_lib)
    print("Loaded extension libraries: ", ext_lib)
    extra_files = {}
    if ext_lib:
        extra_files['extension_libs'] = ext_lib
    return extra_files

def trace_aimnet2_model(model):
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.09572, 0.0, 0.0], [-0.02399, 0.0927, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.int64, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device)
    pbc = torch.tensor([True, True, True], device=device)
    return torch.jit.trace(
        model,
        (positions, atomic_numbers, cell, pbc),
        strict=False,
        check_trace=False,
    )

def download_nnpot_model(model_name):
    os.makedirs("./models", exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", os.path.abspath("./models/warp-cache"))
    os.environ.setdefault("AIMNET_CACHE_DIR", os.path.abspath("./models/aimnet-cache"))
    modelfile_path = os.path.join("./models", f"{model_name}.pt")
    is_ani_model = model_name in ["ani1x", "ani2x"]
    is_emle_model = model_name == "ani2x-emle"
    
    if os.path.exists(modelfile_path):
        if not is_cached_nnpot_model_usable(model_name, modelfile_path):
            print(f"Rebuilding {model_name}.")
        else:
            return modelfile_path

    if os.path.exists(modelfile_path):
        return modelfile_path

    # Download the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        if is_ani_model or model_name == "ani2x-emle":
            os.environ["TORCHANI_DISABLE_EXTENSIONS"] = "1"
        if model_name=="ani1x":
            model = GmxANI1xModel(device)
        elif model_name=="ani2x":
            model = GmxANI2xModel(device)
        elif model_name=="aimnet2":
            model = GmxAIMNet2Model(device)
        elif model_name=="ani2x-emle":
            model = GmxANI2xEMLEModel(device)
        else:
            model_size = model_name.split('-')[1]
            model = GmxMACEModel(model_size, device)
    except ImportError as exc:
        torchani_message = get_torchani_install_error_message(exc)
        if torchani_message is not None:
            raise RuntimeError(torchani_message) from exc
        emle_message = get_emle_install_error_message(model_name, exc)
        if emle_message is not None:
            raise RuntimeError(emle_message) from exc
        raise
    
    # Save the model
    extensions = checkExtensions()
    extensions["nnpot_model_config"] = get_expected_nnpot_model_config(model_name)
    if model_name == "aimnet2":
        scripted_model = trace_aimnet2_model(model)
        scripted_model.save(modelfile_path, _extra_files=extensions)
    elif is_emle_model:
        torch.jit.script(model).save(modelfile_path, _extra_files=extensions)
    elif not "mace" in model_name:
        torch.jit.script(model).save(modelfile_path, _extra_files=extensions)
    else:
        # for MACE, we need to use the e3nn scipting function
        scripted_model = script(model)
        scripted_model.save(modelfile_path, _extra_files=extensions)
    print(f"Saved wrapped model to {modelfile_path}.")
    
    return modelfile_path

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
    return f"""
integrator  = md
dt          = {time_step_ps}
nsteps      = {int(time_scale_ps / time_step_ps)}
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {temperature}
constraints = h-bonds
"""

def get_default_npt_equilibration_mdp_file_content(time_scale_ps=1000, time_step_ps=0.002, temperature=300, pressure=1.0, with_ligand=False):
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
tc-grps         = System
tau_t           = 0.1
ref_t           = {temperature}

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

def get_default_prod_md_mdp_file_content(time_scale_ps=1000, time_step_ps=0.002, temperature=300, pressure=1.0, mdp_type="Initial", random_seed=0, with_ligand=False, nnpot_active=False, nnpot_modelfile_path="models/ani2x.pt", nnpot_input_group="Protein", nnpot_model_name="ani2x"):
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
tc-grps         = System
tau_t           = 0.1
ref_t           = {temperature}

; Constraints
constraints     = h-bonds
constraint_algorithm = lincs
"""
    if nnpot_active:
        content = content + """
; Pressure coupling
; NNPot wrappers currently return energies, not virials/stress. Keep the box
; fixed after NPT equilibration to avoid unstable pressure scaling.
pcoupl          = no
"""
    else:
        content = content + f"""
; Pressure coupling
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {pressure}
compressibility = 4.5e-5
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

    if nnpot_active:
        content = content + "\n; Neural network potential (machine learning interatomic potential)\n"
        content = content + "nnpot-active          = true\n"
        content = content + f"nnpot-modelfile       = {nnpot_modelfile_path}\n"
        content = content + f"nnpot-input-group     = {nnpot_input_group}\n"
        content = content + """nnpot-model-input1    = atom-positions
nnpot-model-input2    = atom-numbers
nnpot-model-input3    = box
nnpot-model-input4    = pbc"""

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
