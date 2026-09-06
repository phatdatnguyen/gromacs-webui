"""Tests for optional neural-potential selection and GROMACS input contracts."""

from __future__ import annotations

import ast
import os
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import utils


class NNPotAvailabilityTests(unittest.TestCase):
    @staticmethod
    def _find_spec_with(installed: set[str]):
        return lambda name: object() if name in installed else None

    def test_dependencies_are_checked_for_the_selected_model(self):
        with mock.patch.object(
            utils.importlib.util,
            "find_spec",
            side_effect=self._find_spec_with({"torch", "torchani"}),
        ):
            self.assertEqual(utils.get_missing_nnpot_packages("ani2x"), [])
            self.assertEqual(
                utils.get_missing_nnpot_packages("mace-small"),
                ["mace", "e3nn"],
            )
            self.assertEqual(utils.get_missing_nnpot_packages("aimnet2"), ["aimnet"])

    def test_general_availability_does_not_require_mace_dependencies(self):
        with mock.patch.object(
            utils.importlib.util,
            "find_spec",
            side_effect=self._find_spec_with({"torch"}),
        ), mock.patch.object(utils, "get_gromacs_nnpot_unavailable_reason", return_value=None):
            self.assertTrue(utils.is_nnpot_available())

    def test_gromacs_build_must_have_torch_support(self):
        utils.get_gromacs_nnpot_unavailable_reason.cache_clear()
        self.addCleanup(utils.get_gromacs_nnpot_unavailable_reason.cache_clear)
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="GROMACS version: 2026.3\nTorch support:       disabled\n",
            stderr="",
        )
        with mock.patch.object(utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"), \
             mock.patch.object(utils.subprocess, "run", return_value=result):
            reason = utils.get_gromacs_nnpot_unavailable_reason()

        self.assertIn("Torch support: disabled", reason)

    def test_torch_enabled_gromacs_build_is_accepted(self):
        utils.get_gromacs_nnpot_unavailable_reason.cache_clear()
        self.addCleanup(utils.get_gromacs_nnpot_unavailable_reason.cache_clear)
        result = utils.subprocess.CompletedProcess(
            ["/opt/gromacs/bin/gmx", "--version"],
            0,
            stdout="GROMACS version: 2026.3\nTorch support:       enabled\n",
            stderr="",
        )
        with mock.patch.object(utils.shutil, "which", return_value="/opt/gromacs/bin/gmx"), \
             mock.patch.object(utils.subprocess, "run", return_value=result):
            self.assertIsNone(utils.get_gromacs_nnpot_unavailable_reason())

    def test_untrusted_model_name_is_rejected_before_it_becomes_a_path(self):
        with self.assertRaisesRegex(ValueError, "Unsupported NNPot model"):
            utils.download_nnpot_model("../../outside")

    def test_corrupt_torchscript_cache_is_quarantined_for_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "ani2x.pt")
            Path(model_path).write_bytes(b"partial archive")
            fake_torch = mock.Mock()
            fake_torch.jit.load.side_effect = RuntimeError(
                "PytorchStreamReader failed reading zip archive: failed finding central directory"
            )
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                usable = utils.is_cached_nnpot_model_usable("ani2x", model_path)

            self.assertFalse(usable)
            self.assertFalse(os.path.exists(model_path))
            self.assertTrue(os.path.exists(model_path + ".invalid"))

    def test_unknown_torchscript_runtime_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "ani2x.pt")
            Path(model_path).write_bytes(b"model")
            fake_torch = mock.Mock()
            fake_torch.jit.load.side_effect = RuntimeError("missing custom operator foo::bar")
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                with self.assertRaisesRegex(RuntimeError, "missing custom operator"):
                    utils.is_cached_nnpot_model_usable("ani2x", model_path)

            self.assertTrue(os.path.exists(model_path))


class NNPotMdpContractTests(unittest.TestCase):
    def _production_mdp(self, model_name: str) -> str:
        return utils.get_default_prod_md_mdp_file_content(
            nnpot_active=True,
            nnpot_model_name=model_name,
            nnpot_modelfile_path=f"/models/{model_name}.pt",
        )

    def test_mace_uses_gromacs_neighbor_pairs_and_model_cutoff(self):
        content = self._production_mdp("mace-medium")

        self.assertIn("nnpot-model-input3    = nnp-charge", content)
        self.assertIn("nnpot-model-input4    = atom-pairs", content)
        self.assertIn("nnpot-model-input5    = pair-shifts", content)
        self.assertIn("nnpot-model-input6    = box", content)
        self.assertIn("nnpot-model-input7    = pbc", content)
        self.assertIn("pair-cutoff            = 0.5", content)

    def test_charge_aware_models_receive_topology_charge(self):
        content = self._production_mdp("aimnet2")
        self.assertIn("nnpot-model-input3    = nnp-charge", content)
        self.assertIn("nnpot-model-input4    = box", content)
        self.assertIn("nnpot-model-input5    = pbc", content)
        self.assertNotIn("atom-pairs", content)

    def test_emle_uses_electrostatic_embedding_and_mm_environment(self):
        content = self._production_mdp("ani2x-emle")

        self.assertIn("nnpot-embedding       = electrostatic-model", content)
        self.assertIn("nnpot-model-input3    = atom-positions-mm", content)
        self.assertIn("nnpot-model-input4    = atom-charges-mm", content)
        self.assertIn("nnpot-model-input5    = nnp-charge", content)
        self.assertIn("nnpot-model-input6    = box", content)

    def test_ani_contract_checks_charge_then_passes_box_and_pbc(self):
        content = self._production_mdp("ani2x")

        self.assertIn("nnpot-model-input3    = nnp-charge", content)
        self.assertIn("nnpot-model-input4    = box", content)
        self.assertIn("nnpot-model-input5    = pbc", content)
        self.assertNotIn("atom-pairs", content)

    def test_unknown_model_cannot_generate_an_mdp(self):
        with self.assertRaisesRegex(ValueError, "Unsupported NNPot model"):
            self._production_mdp("not-a-model")

    def test_cache_fingerprints_invalidate_old_wrapper_contracts(self):
        self.assertIn("nonmutating-box", utils.get_expected_nnpot_model_config("ani2x"))
        self.assertIn("gromacs-pairs", utils.get_expected_nnpot_model_config("mace-small"))
        self.assertIn("runtime-charge", utils.get_expected_nnpot_model_config("aimnet2"))
        self.assertIn("runtime-charge", utils.get_expected_nnpot_model_config("ani2x-emle"))


class NNPotWrapperSignatureTests(unittest.TestCase):
    """Keep MDP input order exactly synchronized with TorchScript forward args.

    The optional ML stack is intentionally absent from the normal test
    environment, so parse the wrapper source without importing torch.
    """

    @classmethod
    def setUpClass(cls):
        source_path = Path(__file__).resolve().parents[1] / "nnpot_models.py"
        cls.tree = ast.parse(source_path.read_text(encoding="utf-8"))

    def _forward_arguments(self, class_name: str) -> list[str]:
        class_node = next(
            node for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        forward = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        return [argument.arg for argument in forward.args.args[1:]]

    @staticmethod
    def _mdp_inputs(model_name: str) -> list[str]:
        section = utils.get_nnpot_model_input_mdp_section(model_name)
        numbered_inputs = []
        for line in section.splitlines():
            match = re.match(r"nnpot-model-input(\d+)\s*=\s*(\S+)", line)
            if match:
                numbered_inputs.append((int(match.group(1)), match.group(2)))
        return [value for _, value in sorted(numbered_inputs)]

    def test_every_wrapper_signature_matches_its_mdp_input_order(self):
        contracts = {
            "ani1x": (
                "GmxANI1xModel",
                ["positions", "atomic_numbers", "nnp_charge", "box", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "box", "pbc"],
            ),
            "ani2x": (
                "GmxANI2xModel",
                ["positions", "atomic_numbers", "nnp_charge", "box", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "box", "pbc"],
            ),
            "mace-medium": (
                "GmxMACEModel",
                ["positions", "atomic_numbers", "nnp_charge", "pairs", "shifts", "cell", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "atom-pairs", "pair-shifts", "box", "pbc"],
            ),
            "aimnet2": (
                "GmxAIMNet2Model",
                ["positions", "atomic_numbers", "nnp_charge", "cell", "pbc"],
                ["atom-positions", "atom-numbers", "nnp-charge", "box", "pbc"],
            ),
            "ani2x-emle": (
                "GmxANI2xEMLEModel",
                ["positions_nn", "atomic_numbers", "positions_mm", "charges_mm", "nnp_charge", "cell"],
                ["atom-positions", "atom-numbers", "atom-positions-mm", "atom-charges-mm", "nnp-charge", "box"],
            ),
        }

        for model_name, (class_name, wrapper_args, mdp_inputs) in contracts.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(self._forward_arguments(class_name), wrapper_args)
                self.assertEqual(self._mdp_inputs(model_name), mdp_inputs)


if __name__ == "__main__":
    unittest.main()
