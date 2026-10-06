"""TorchScript regressions that do not require optional model packages."""

from __future__ import annotations

import unittest
from typing import Optional

try:
    import torch
except ImportError:  # Torch is optional for the ordinary WebUI test install.
    torch = None


if torch is not None:
    class _DeviceState(torch.nn.Module):
        """Minimal nested module matching EMLE's persisted device attributes."""

        def __init__(self, with_base: bool = False) -> None:
            super().__init__()
            self._device = torch.device("cpu")
            if with_base:
                self._emle_base = _DeviceState()


    class _ScalarChargeEMLE(torch.nn.Module):
        """Small stand-in with the same scalar-charge contract as ANI2xEMLE."""

        def __init__(self) -> None:
            super().__init__()
            self._device = torch.device("cpu")
            self._emle = _DeviceState(with_base=True)

        def forward(
            self,
            atomic_numbers: torch.Tensor,
            charges_mm: torch.Tensor,
            positions_nn: torch.Tensor,
            positions_mm: torch.Tensor,
            cell: Optional[torch.Tensor],
            qm_charge: int,
        ) -> torch.Tensor:
            del atomic_numbers, charges_mm, cell
            return (
                positions_nn.sum().reshape(1)
                + positions_mm.sum().reshape(1)
                + float(qm_charge)
            )


    class _CapturingMACECore(torch.nn.Module):
        """Record the graph handed to MACE without loading a foundation model."""

        def __init__(self) -> None:
            super().__init__()
            self.last_data = None

        def forward(self, data, **_kwargs):
            self.last_data = data
            return {"energy": data["positions"].sum() * 0.0}


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EMLETorchScriptContractTests(unittest.TestCase):
    @staticmethod
    def _scripted_wrapper():
        from nnpot_models import GmxANI2xEMLEModel

        wrapper = GmxANI2xEMLEModel.__new__(GmxANI2xEMLEModel)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = _ScalarChargeEMLE()
        wrapper.is_nnpops = False
        wrapper.register_buffer(
            "supported_atomic_numbers",
            torch.tensor([1, 6, 7, 8, 16], dtype=torch.int64),
        )
        wrapper.length_conversion = 10.0
        wrapper.energy_conversion = 1.0
        return torch.jit.script(wrapper)

    @staticmethod
    def _inputs(charge: float):
        return (
            torch.zeros((2, 3), dtype=torch.float64, requires_grad=True),
            torch.tensor([1, 8], dtype=torch.int64),
            torch.zeros((1, 3), dtype=torch.float64, requires_grad=True),
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([charge], dtype=torch.float64),
            torch.eye(3, dtype=torch.float64),
        )

    def test_one_dimensional_neutral_charge_reaches_scalar_emle_argument(self):
        energy, forces_nn, forces_mm = self._scripted_wrapper()(
            *self._inputs(0.0)
        )

        self.assertEqual(energy.item(), 0.0)
        self.assertEqual(tuple(forces_nn.shape), (2, 3))
        self.assertEqual(tuple(forces_mm.shape), (1, 3))

    def test_integral_nonzero_charge_is_rejected_inside_wrapper(self):
        with self.assertRaisesRegex(Exception, "neutral NNP regions only"):
            self._scripted_wrapper()(*self._inputs(1.0))

    def test_fractional_charge_is_rejected_before_emle_call(self):
        with self.assertRaisesRegex(Exception, "integer NNP-region charge"):
            self._scripted_wrapper()(*self._inputs(0.5))

    def test_unsupported_element_is_rejected_inside_scripted_wrapper(self):
        inputs = list(self._inputs(0.0))
        inputs[1] = torch.tensor([1, 9], dtype=torch.int64)

        with self.assertRaisesRegex(Exception, "does not support"):
            self._scripted_wrapper()(*inputs)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MACEPairInputContractTests(unittest.TestCase):
    @staticmethod
    def _wrapper():
        from nnpot_models import GmxMACEModel

        wrapper = GmxMACEModel.__new__(GmxMACEModel)
        torch.nn.Module.__init__(wrapper)
        core = _CapturingMACECore()
        wrapper.model = core
        wrapper.register_buffer(
            "atomic_number_table", torch.tensor([1, 6], dtype=torch.int64))
        wrapper.register_buffer("r_max", torch.tensor(5.0))
        wrapper.length_conversion = 10.0
        wrapper.energy_conversion = 1.0
        return wrapper, core

    def test_gromacs_half_pairs_become_bidirectional_mace_edges(self):
        wrapper, core = self._wrapper()
        wrapper(
            torch.tensor([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]),
            torch.tensor([1, 6]),
            torch.tensor([0.0]),
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.eye(3),
            torch.tensor([True, True, True]),
        )

        graph = core.last_data
        self.assertIsNotNone(graph)
        self.assertTrue(torch.equal(
            graph["edge_index"],
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        ))
        # GROMACS shifts the first atom of each half-list pair; MACE applies a
        # shift to the receiver in r_j-r_i+shift.  The forward edge therefore
        # needs the opposite sign and the generated reverse edge the original.
        self.assertTrue(torch.equal(
            graph["shifts"],
            torch.tensor([
                [-10.0, -20.0, -30.0],
                [10.0, 20.0, 30.0],
            ]),
        ))

    def test_mace_rejects_out_of_range_pair_indices(self):
        wrapper, _ = self._wrapper()
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            wrapper(
                torch.zeros((2, 3)), torch.tensor([1, 6]),
                torch.tensor([0.0]),
                torch.tensor([[0, 2]], dtype=torch.int64),
                torch.zeros((1, 3)), torch.eye(3),
                torch.tensor([True, True, True]),
            )


if __name__ == "__main__":
    unittest.main()
