"""Generate a TorchScript neural network potential (MLIP) model for GROMACS.

The resulting ``.pt`` file can be referenced from a production ``.mdp`` file via
``nnpot-modelfile`` to run MD with a machine learning interatomic potential.

Usage:
    python generate_nnpot_model.py [save_path] [model_name]

    save_path   Output path for the TorchScript model (default: ani2x.pt)
    model_name  Pre-trained model to wrap (default: ANI2x)

Requirements:
    pip install torch torchani
"""

import sys

import torch
from torch import nn
from typing import Optional

class GmxNNPotModelWrapper(nn.Module):
    def __init__(self, model_name="ANI2x"):
        super().__init__()

        # Load a pre-trained model from TorchANI
        if model_name == "ANI2x":
            from torchani.models import ANI2x
            self.model = ANI2x(periodic_table_index=True)
        elif model_name == "ANI1x":
            from torchani.models import ANI1x
            self.model = ANI1x(periodic_table_index=True)
        elif model_name == "ANI1ccx":
            from torchani.models import ANI1ccx
            self.model = ANI1ccx(periodic_table_index=True)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # GROMACS and TorchANI use different unit conventions
        self.length_conversion = 10.0   # nm --> Å
        self.energy_conversion = 2625.5  # Hartree --> kJ/mol

    def forward(self, positions, atomic_numbers,
                box: Optional[torch.Tensor] = None, pbc: Optional[torch.Tensor] = None):

        # Prepare the inputs for the model
        atomic_numbers = atomic_numbers.unsqueeze(0)
        positions = positions.unsqueeze(0) * self.length_conversion
        if box is not None:
            box *= self.length_conversion

        # Forward pass
        result = self.model((atomic_numbers, positions), box, pbc)

        energy = result.energies[0] * self.energy_conversion

        return energy


def main():
    save_path = sys.argv[1] if len(sys.argv) > 1 else "ani2x.pt"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ANI2x"

    print(f"Building {model_name} model wrapper...")
    model = GmxNNPotModelWrapper(model_name=model_name)

    print(f"Compiling with torch.jit.script and saving to: {save_path}")
    torch.jit.script(model).save(save_path)

    print("Done.")


if __name__ == "__main__":
    main()
