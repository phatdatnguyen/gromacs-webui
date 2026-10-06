"""Regression tests for AMBER's warning-only cut-off electrostatics policy."""

from __future__ import annotations

import contextlib
import io
import unittest.mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow

from .testing_support import WorkingDirectoryTestCase


class AmberCutoffHandlerWarningTests(WorkingDirectoryTestCase):
    """Every GROMPP entry point must surface the expert AMBER choice."""

    GROMPP_HANDLERS = (
        "on_generate_ions_tpr_file",
        "on_generate_energy_minimization_tpr_file",
        "on_generate_nvt_equilibration_tpr_file",
        "on_generate_npt_equilibration_tpr_file",
        "on_generate_prod_md_tpr_file",
    )

    def setUp(self) -> None:
        super().setUp()
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "amber99sb-ildn.ff/forcefield.itp"\n')
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "integrator = steep\ncutoff-scheme = Verlet\n"
                "rlist = 1.0\nrvdw = 1.0\nrcoulomb = 1.0\n"
                "coulombtype = Cut-off\nDispCorr = EnerPres\n"
            )

    def test_warning_reaches_all_grompp_handlers_in_both_workflows(self):
        arguments = (
            "input.gro", "topol.top", "step.mdp", "step.tpr", 0,
            "AMBER99SB-ILDN",
        )
        for module in (protein_workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(
                            module, "run_checked_command") as run, \
                        contextlib.redirect_stdout(io.StringIO()) as terminal:
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path, *arguments)

                run.assert_called_once()
                plain_status = self.plain_text(status)
                self.assertIn("color:orange", status)
                self.assertIn("generated successfully", plain_status)
                self.assertIn("coulombtype=Cut-off", plain_status)
                self.assertIn("allowed", plain_status.lower())
                self.assertIn("WARNING:", terminal.getvalue())
                self.assertIn("coulombtype=Cut-off", terminal.getvalue())

    def test_pme_with_enerpres_keeps_the_existing_green_success_path(self):
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "integrator = steep\ncutoff-scheme = Verlet\n"
                "rlist = 1.0\nrvdw = 1.0\nrcoulomb = 1.0\n"
                "coulombtype = PME\nDispCorr = EnerPres\n"
            )
        # NPT and production only add their independent continuation warning
        # when the input structure has no matching checkpoint.
        with open(self.path("input.cpt"), "w") as handle:
            handle.write("fixture")

        arguments = (
            "input.gro", "topol.top", "step.mdp", "step.tpr", 0,
            "AMBER99SB-ILDN",
        )
        for module in (protein_workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(
                            module, "run_checked_command") as run, \
                        contextlib.redirect_stdout(io.StringIO()) as terminal:
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path, *arguments)

                run.assert_called_once()
                self.assertIn("color:green", status)
                self.assertIn("generated successfully", self.plain_text(status))
                self.assertNotIn("coulombtype=Cut-off", status)
                self.assertNotIn("WARNING:", terminal.getvalue())
