import torch
from torch import nn
from typing import Optional
import os

class ANI1xModelWrapper(nn.Module):
    def __init__(self, device:str):
        super().__init__()
        from torchani.models import ANI1x

        # Load a pre-trained ANI-1x model using pure PyTorch AEV so the saved
        # TorchScript model does not depend on optional cuAEV custom classes.
        self.model = ANI1x(
            periodic_table_index=True,
            neighborlist="adaptive",
            strategy="pyaev",
            device=device,
        )

        # GROMACS and TorchANI use different unit conventions
        self.length_conversion = 10.0   # nm --> Å
        self.energy_conversion = 2625.5 # Hartree --> kJ/mol

    def forward(self, positions, atomic_numbers,
                box: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):

        # Prepare the inputs for the model
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if box is not None:
            box *= self.length_conversion

        # Forward pass
        result = self.model((atomic_numbers, positions), box, pbc)

        energy = result.energies[0] * self.energy_conversion

        return energy

class ANI2xModelWrapper(nn.Module):
    def __init__(self, device:str):
        super().__init__()
        from torchani.models import ANI2x

        # Load a pre-trained ANI-2x model
        self.model = ANI2x(
            periodic_table_index=True,
            neighborlist="adaptive",
            strategy="pyaev",
            device=device,
        )

        # GROMACS and TorchANI use different unit conventions
        self.length_conversion = 10.0   # nm --> Å
        self.energy_conversion = 2625.5 # Hartree --> kJ/mol

    def forward(self, positions, atomic_numbers,
                box: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):

        # Prepare the inputs for the model
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if box is not None:
            box *= self.length_conversion

        # Forward pass
        result = self.model((atomic_numbers, positions), box, pbc)

        energy = result.energies[0] * self.energy_conversion

        return energy

class GmxMACEModel(torch.nn.Module):
    def __init__(self, size: str, device:str, **kwargs):
        super().__init__()
        from mace.calculators import mace_off

        assert size in ["small", "medium", "large"], "Invalid MACE model size"
        model = mace_off(size, device, return_raw_model=True).to(torch.float32)
        self.model = model
        self.z_table = model.atomic_numbers.tolist()
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        self.length_conversion = 10.0       # nm (gmx) --> Å (mace)
        self.energy_conversion = 96.4853    # eV (mace) --> kJ/mol (gmx)
    
    def forward(self, positions, atomic_numbers,
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):
        
        # Prepare the model input
        positions = positions * self.length_conversion
        n_atoms = positions.shape[0]
        device = positions.device
        if cell is not None:
            cell = cell * self.length_conversion
        else:
            cell = torch.zeros(3, 3).to(device)

        # Build a directed neighbor list inside the scripted model. This follows
        # the no-pairs wrapper style from gmx-nnpot-tools and avoids depending on
        # GROMACS atom-pairs/pair-shifts inputs during grompp model checks.
        atom_indices = torch.arange(n_atoms, dtype=torch.int64, device=device)
        src_all = atom_indices.repeat_interleave(n_atoms)
        dst_all = atom_indices.repeat(n_atoms)
        non_self = src_all != dst_all
        src_all = src_all[non_self]
        dst_all = dst_all[non_self]

        deltas = positions[src_all] - positions[dst_all]
        use_pbc = False
        if cell is not None:
            use_pbc = bool(torch.abs(torch.linalg.det(cell)) > 1.0e-8)
            if pbc is not None:
                use_pbc = use_pbc and bool(torch.any(pbc))

        if use_pbc:
            shift_indices = torch.round(torch.mm(deltas, torch.linalg.inv(cell)))
            shifts_all = torch.mm(shift_indices, cell)
            wrapped_deltas = deltas - shifts_all
        else:
            shifts_all = torch.zeros((deltas.shape[0], 3), dtype=positions.dtype, device=device)
            wrapped_deltas = deltas

        within_cutoff = torch.linalg.norm(wrapped_deltas, dim=1) < self.r_max
        pairs = torch.stack([src_all[within_cutoff], dst_all[within_cutoff]], dim=0)
        shifts = shifts_all[within_cutoff]

        # One hot encoding of atomic numbers
        # no need to convert since gromacs and mace use the same atomic numbers
        nodeAttrs = torch.zeros(n_atoms, len(self.z_table), device=device)
        indices = torch.stack([torch.tensor(self.z_table.index(z)) for z in atomic_numbers])
        nodeAttrs[torch.arange(n_atoms), indices] = 1.0

        # other inputs
        ptr = torch.tensor([0, n_atoms], dtype=torch.int64, requires_grad=False, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.int64, device=device)
        if pbc is None:
            pbc = torch.tensor([True, True, True], requires_grad=False, device=device)
        
        # Prepare the input dict
        input_data = {
            "ptr": ptr,
            "node_attrs": nodeAttrs,
            "batch": batch,
            "positions": positions,
            "edge_index": pairs,
            "shifts": shifts,
            "pbc": pbc,
            "cell": cell,
        }

        # run the model
        out = self.model(
            input_data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False
        )

        total_energy = out["energy"]
        if total_energy is None:
            total_energy = torch.tensor(0.0, device=device)

        return total_energy * self.energy_conversion

class GmxAIMNet2Model(torch.nn.Module):
    def __init__(self, device: str, charge=0, mult=1, **kwargs):
        super().__init__()
        os.environ.setdefault("WARP_CACHE_PATH", os.path.abspath("./models/warp-cache"))
        os.environ.setdefault("AIMNET_CACHE_DIR", os.path.abspath("./models/aimnet-cache"))
        from aimnet.calculators.model_registry import get_model_path
        from aimnet.models.base import load_model

        model_path = get_model_path("aimnet2")
        self.model, _ = load_model(model_path, device=device)
        self.model = self.model.double()
        self.register_buffer("charge", torch.tensor([charge], dtype=torch.float64, device=device))
        self.register_buffer("mult", torch.tensor([mult], dtype=torch.int64, device=device))
        self.length_conversion = 10.0       # nm (gmx) --> Å (aimnet)
        self.energy_conversion = 96.4853    # eV (aimnet) --> kJ/mol (gmx)

    def forward(self, positions, atomic_numbers,
                cell: Optional[torch.Tensor]=None, pbc: Optional[torch.Tensor]=None):
        # Prepare the model input
        positions = positions.to(torch.float64) * self.length_conversion
        atomic_numbers = atomic_numbers.to(dtype=torch.int64, device=positions.device)
        if cell is not None:
            cell = cell.to(dtype=torch.float64, device=positions.device) * self.length_conversion
        else:
            cell = torch.zeros(3, 3, dtype=torch.float64, device=positions.device)
        if pbc is not None:
            pbc = pbc.to(device=positions.device)
        else:
            pbc = torch.tensor([False, False, False], dtype=torch.bool, device=positions.device)

        # Prepare input for aimnet model
        input_data = {
            "coord": positions.unsqueeze(0),
            "numbers": atomic_numbers.unsqueeze(0),
            "mol_idx": torch.zeros(atomic_numbers.shape[0], dtype=torch.int64, device=positions.device).unsqueeze(0),
            "charge": self.charge,
            "mult": self.mult,
            "cell": cell.unsqueeze(0),
            "pbc": pbc.unsqueeze(0),
        }

        result = self.model(input_data)

        energy = result["energy"].reshape(-1)[0] * self.energy_conversion

        return energy
