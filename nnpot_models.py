"""GROMACS-facing wrappers around neural-network potentials.

Each wrapper converts between GROMACS units (nm, kJ/mol) and the model's own
convention, and exposes a forward signature that the GROMACS neural-network
potential interface can call.
"""

import os
import torch
from torch import nn
from typing import Optional, Tuple

from path_security import MODEL_ROOT

def load_emle_model_classes() -> tuple[type, float, float]:
    """Import EMLE behind a TorchANI 2.8 shim and return its class and unit factors."""
    # EMLE currently imports SpeciesEnergies from the TorchANI 2.7 location.
    # TorchANI 2.8 moved it to torchani.tuples, so provide the old attribute
    # before importing EMLE.
    import torchani.aev as torchani_aev
    import torchani.nn as torchani_nn
    from torchani.tuples import SpeciesAEV, SpeciesEnergies

    if not hasattr(torchani_nn, "SpeciesEnergies"):
        torchani_nn.SpeciesEnergies = SpeciesEnergies
    if not hasattr(torchani_aev, "SpeciesAEV"):
        torchani_aev.SpeciesAEV = SpeciesAEV

    from emle.models import ANI2xEMLE
    from emle._units import _HARTREE_TO_KJ_MOL, _NANOMETER_TO_ANGSTROM

    def _add_torchani28_hook(self) -> None:
        """Capture AEV output via a forward hook, as TorchANI 2.8 no longer stores it."""
        from torch import Tensor
        from typing import Optional, Tuple

        self._ani2x.aev_computer._aev = torch.empty(0, device=self._device)

        def hook(
            module,
            input: Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]],
            output: Tensor,
        ):
            module._aev = output

        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook)

    ANI2xEMLE._add_hook = _add_torchani28_hook

    return ANI2xEMLE, _NANOMETER_TO_ANGSTROM, _HARTREE_TO_KJ_MOL

class GmxANI1xModel(nn.Module):
    """ANI-1x wrapped for GROMACS, using pure PyTorch AEV."""
    def __init__(self, device: str) -> None:
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

    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor,
                nnp_charge: torch.Tensor,
                box: Optional[torch.Tensor] = None,
                pbc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the potential energy in kJ/mol for GROMACS coordinates given in nm."""

        if nnp_charge.numel() != 1:
            raise RuntimeError("ANI-1x requires one total NNP-region charge")
        if bool(torch.abs(nnp_charge.reshape(-1)[0]) > 1.0e-4):
            raise RuntimeError("ANI-1x supports neutral NNP regions only")

        # Prepare the inputs for the model
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if box is not None:
            box = box * self.length_conversion

        # Forward pass
        result = self.model((atomic_numbers, positions), box, pbc)

        energy = result.energies[0] * self.energy_conversion

        return energy

class GmxANI2xModel(nn.Module):
    """ANI-2x wrapped for GROMACS, using pure PyTorch AEV."""
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

    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor,
                nnp_charge: torch.Tensor,
                box: Optional[torch.Tensor] = None,
                pbc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the potential energy in kJ/mol for GROMACS coordinates given in nm."""

        if nnp_charge.numel() != 1:
            raise RuntimeError("ANI-2x requires one total NNP-region charge")
        if bool(torch.abs(nnp_charge.reshape(-1)[0]) > 1.0e-4):
            raise RuntimeError("ANI-2x supports neutral NNP regions only")

        # Prepare the inputs for the model
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if box is not None:
            box = box * self.length_conversion

        # Forward pass
        result = self.model((atomic_numbers, positions), box, pbc)

        energy = result.energies[0] * self.energy_conversion

        return energy

class GmxMACEModel(torch.nn.Module):
    """A MACE foundation model wrapped for GROMACS."""
    def __init__(self, size: str, device: str, **kwargs: object) -> None:
        super().__init__()
        from mace.calculators import mace_off

        if size not in ["small", "medium", "large"]:
            raise ValueError("Invalid MACE model size")
        model = mace_off(size, device, return_raw_model=True).to(torch.float32)
        self.model = model
        self.z_table = model.atomic_numbers.tolist()
        self.register_buffer(
            "atomic_number_table",
            torch.tensor(self.z_table, dtype=torch.int64, device=device),
        )
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
    
    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor,
                nnp_charge: torch.Tensor, pairs: torch.Tensor, shifts: torch.Tensor,
                cell: Optional[torch.Tensor] = None,
                pbc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the potential energy in kJ/mol for GROMACS coordinates given in nm."""

        if nnp_charge.numel() != 1:
            raise RuntimeError("MACE-OFF requires one total NNP-region charge")
        if bool(torch.abs(nnp_charge.reshape(-1)[0]) > 1.0e-4):
            raise RuntimeError("MACE-OFF supports neutral NNP regions only")

        # Prepare the model input
        positions = positions.to(dtype=self.r_max.dtype) * self.length_conversion
        n_atoms = positions.shape[0]
        device = positions.device
        if cell is not None:
            cell = cell.to(dtype=positions.dtype, device=device) * self.length_conversion
        else:
            cell = torch.zeros(3, 3, dtype=positions.dtype, device=device)

        # GROMACS already owns a scalable, PBC-aware neighbour list.  Consuming
        # atom-pairs and pair-shifts here also handles triclinic/partial-PBC boxes
        # and periodic images correctly; the former all-pairs/minimum-image code
        # was both O(N^2) and physically incomplete.
        if pairs.dim() != 2 or pairs.shape[1] != 2:
            raise RuntimeError("MACE atom-pairs must have shape (N_pairs, 2)")
        if shifts.dim() != 2 or shifts.shape[1] != 3 or shifts.shape[0] != pairs.shape[0]:
            raise RuntimeError("MACE pair-shifts must have shape (N_pairs, 3)")
        pairs = pairs.to(dtype=torch.int64, device=device)
        if pairs.numel() > 0 and bool(torch.any((pairs < 0) | (pairs >= n_atoms))):
            raise RuntimeError("MACE atom-pairs contain an out-of-range atom index")
        reverse_pairs = pairs[:, [1, 0]]
        pairs = torch.cat([pairs, reverse_pairs], dim=0).t()
        shifts = shifts.to(dtype=positions.dtype, device=device)
        shifts = torch.cat([-shifts, shifts], dim=0) * self.length_conversion

        # One hot encoding of atomic numbers
        # (GROMACS and MACE use the same atomic-number convention).
        atomic_numbers = atomic_numbers.to(dtype=torch.int64, device=device)
        if atomic_numbers.numel() != n_atoms:
            raise RuntimeError("MACE requires one atomic number per position")
        matches = atomic_numbers.reshape(-1, 1) == self.atomic_number_table.reshape(1, -1)
        if not bool(torch.all(torch.sum(matches, dim=1) == 1)):
            raise RuntimeError("The selected MACE model does not support one or more elements")
        nodeAttrs = matches.to(dtype=positions.dtype)

        # other inputs
        ptr = torch.tensor([0, n_atoms], dtype=torch.int64, requires_grad=False, device=device)
        batch = torch.zeros(n_atoms, dtype=torch.int64, device=device)
        if pbc is None:
            pbc = torch.tensor([False, False, False], requires_grad=False, device=device)
        else:
            pbc = pbc.to(dtype=torch.bool, device=device)
        
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
    """AIMNet2 wrapped for GROMACS; traced rather than scripted."""
    def __init__(self, device: str, mult: int = 1, **kwargs: object) -> None:
        super().__init__()
        os.environ.setdefault("WARP_CACHE_PATH", str(MODEL_ROOT / "warp-cache"))
        os.environ.setdefault("AIMNET_CACHE_DIR", str(MODEL_ROOT / "aimnet-cache"))
        from aimnet.calculators.model_registry import get_model_path
        from aimnet.models.base import load_model

        model_path = get_model_path("aimnet2")
        self.model, _ = load_model(model_path, device=device)
        self.model = self.model.double()
        self.register_buffer("mult", torch.tensor([mult], dtype=torch.int64, device=device))
        self.length_conversion = 10.0       # nm (gmx) --> Å (aimnet)
        self.energy_conversion = 96.4853    # eV (aimnet) --> kJ/mol (gmx)

    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor,
                nnp_charge: torch.Tensor,
                cell: Optional[torch.Tensor] = None,
                pbc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the potential energy in kJ/mol for GROMACS coordinates given in nm."""
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
        if nnp_charge.numel() != 1:
            raise RuntimeError("AIMNet2 requires one total NNP-region charge")
        charge = nnp_charge.to(dtype=torch.float64, device=positions.device).reshape(1)

        # Prepare input for aimnet model
        input_data = {
            "coord": positions.unsqueeze(0),
            "numbers": atomic_numbers.unsqueeze(0),
            "mol_idx": torch.zeros(atomic_numbers.shape[0], dtype=torch.int64, device=positions.device).unsqueeze(0),
            "charge": charge,
            "mult": self.mult,
            "cell": cell.unsqueeze(0),
            "pbc": pbc.unsqueeze(0),
        }

        result = self.model(input_data)

        energy = result["energy"].reshape(-1)[0] * self.energy_conversion

        return energy

class GmxANI2xEMLEModel(torch.nn.Module):
    """ANI-2x with EMLE embedding, wrapped for GROMACS."""
    def __init__(self, device: str, **kwargs: object) -> None:
        super().__init__()
        ANI2xEMLE, length_conversion, energy_conversion = load_emle_model_classes()
        kwargs.setdefault("device", torch.device(device))
        self.model = ANI2xEMLE(**kwargs)
        self.is_nnpops = self.model._is_nnpops

        self.length_conversion = length_conversion
        self.energy_conversion = energy_conversion

    def forward(self, positions_nn: torch.Tensor, atomic_numbers: torch.Tensor,
                positions_mm: torch.Tensor, charges_mm: torch.Tensor,
                nnp_charge: torch.Tensor,
                cell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return energy plus NNP/MM forces for electrostatic EMLE embedding."""
        device = positions_nn.device
        # convert units
        positions_nn = positions_nn * self.length_conversion
        positions_mm = positions_mm.to(dtype=positions_nn.dtype, device=device) * self.length_conversion
        if cell is not None:
            cell = cell.to(dtype=positions_nn.dtype, device=device) * self.length_conversion
        charges_mm = charges_mm.to(dtype=positions_nn.dtype, device=device)
        atomic_numbers = atomic_numbers.to(dtype=torch.int64, device=device)
        if nnp_charge.numel() != 1:
            raise RuntimeError("ANI2x-EMLE requires one total NNP-region charge")
        qm_charge = nnp_charge.to(dtype=positions_nn.dtype, device=device).reshape(1)

        if not self.is_nnpops:
            positions_nn = positions_nn.unsqueeze(0)
            positions_mm = positions_mm.unsqueeze(0)
            atomic_numbers = atomic_numbers.unsqueeze(0)
            charges_mm = charges_mm.unsqueeze(0)

        E = self.model(atomic_numbers, charges_mm, positions_nn, positions_mm, cell, qm_charge)
        E_tot = E.sum() * self.energy_conversion

        # Electrostatic embedding depends on both coordinate sets.  GROMACS can
        # differentiate the first model input itself, but requires explicit MM
        # forces as additional outputs.
        gradients = torch.autograd.grad(
            [E_tot], [positions_nn, positions_mm], allow_unused=True
        )
        gradient_nn, gradient_mm = gradients
        if gradient_nn is None:
            forces_nn = torch.zeros_like(positions_nn, device=device)
        else:
            forces_nn = -gradient_nn * self.length_conversion
        if gradient_mm is None:
            forces_mm = torch.zeros_like(positions_mm, device=device)
        else:
            forces_mm = -gradient_mm * self.length_conversion

        if not self.is_nnpops:
            forces_nn = forces_nn.squeeze(0)
            forces_mm = forces_mm.squeeze(0)

        return E_tot, forces_nn, forces_mm
