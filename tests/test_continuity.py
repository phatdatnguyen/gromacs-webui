"""Regression tests for carrying equilibration state between GROMACS stages."""

from __future__ import annotations

import os
import unittest.mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow
import utils
from .testing_support import WorkingDirectoryTestCase


class MdpContinuityTests(WorkingDirectoryTestCase):
    def setUp(self):
        super().setUp()
        # The command runner is mocked, but the warning policy deliberately
        # inspects the validated topology to decide whether GROMOS needs its
        # narrowly scoped compatibility allowance.
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "amber99sb-ildn.ff/forcefield.itp"\n')

    def test_only_nvt_generates_fresh_velocities(self):
        nvt = utils.get_default_nvt_equilibration_mdp_file_content()
        npt = utils.get_default_npt_equilibration_mdp_file_content()
        production = utils.get_default_prod_md_mdp_file_content(mdp_type="Initial", random_seed=42)

        self.assertIn("continuation = no", nvt)
        self.assertIn("gen_vel     = yes", nvt)
        for content in (npt, production):
            with self.subTest(content=content[:40]):
                self.assertIn("continuation    = yes", content)
                self.assertIn("gen_vel         = no", content)
                self.assertNotIn("gen_seed", content)

    def test_npt_grompp_uses_the_checkpoint_matching_its_input_structure(self):
        with open(self.path("nvt.cpt"), "wb") as handle:
            handle.write(b"checkpoint")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(
                        module, "_validate_grompp_inputs",
                        return_value=(self.path("npt.mdp"), self.path("topol.top"))), \
                    unittest.mock.patch.object(module, "run_checked_command") as runner:
                _, status = module.on_generate_npt_equilibration_tpr_file(
                    self.working_directory_path, "nvt.gro", "topol.top",
                    "npt.mdp", "npt.tpr", 0)

                command = runner.call_args.args[0]
                self.assertEqual(command[command.index("-t") + 1], os.path.abspath(self.path("nvt.cpt")))
                self.assertEqual(runner.call_args.kwargs["cwd"], os.path.abspath(self.working_directory_path))
                self.assertIn("successfully", self.plain_text(status))

    def test_production_grompp_uses_the_npt_checkpoint(self):
        with open(self.path("npt.cpt"), "wb") as handle:
            handle.write(b"checkpoint")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(
                        module, "_validate_grompp_inputs",
                        return_value=(self.path("md.mdp"), self.path("topol.top"))), \
                    unittest.mock.patch.object(module, "run_checked_command") as runner:
                _, status = module.on_generate_prod_md_tpr_file(
                    self.working_directory_path, "npt.gro", "topol.top",
                    "md.mdp", "md.tpr", 0)

                command = runner.call_args.args[0]
                self.assertEqual(command[command.index("-t") + 1], os.path.abspath(self.path("npt.cpt")))
                self.assertEqual(runner.call_args.kwargs["cwd"], os.path.abspath(self.working_directory_path))
                self.assertIn("successfully", self.plain_text(status))

    def test_missing_checkpoint_is_reported_and_not_faked(self):
        with unittest.mock.patch.object(
                protein_workflow, "_validate_grompp_inputs",
                return_value=(self.path("md.mdp"), self.path("topol.top"))), \
                unittest.mock.patch.object(protein_workflow, "run_checked_command") as runner:
            _, status = protein_workflow.on_generate_prod_md_tpr_file(
                self.working_directory_path, "npt.gro", "topol.top",
                "md.mdp", "md.tpr", 0)

        self.assertNotIn("-t", runner.call_args.args[0])
        self.assertIn("No matching checkpoint", self.plain_text(status))


class ProductionResumeTests(WorkingDirectoryTestCase):
    def write_resume_pair(self, stem="md"):
        for suffix in (".tpr", ".cpt"):
            with open(self.path(stem + suffix), "wb") as handle:
                handle.write(b"test")

    def test_matching_resume_pair_is_accepted(self):
        self.write_resume_pair()

        self.assertEqual(
            utils.require_matching_resume_files(
                self.working_directory_path, "md.tpr", "md.cpt"),
            ("md.tpr", "md.cpt"),
        )

    def test_foreign_or_missing_checkpoint_is_rejected(self):
        self.write_resume_pair()
        with open(self.path("other.cpt"), "wb") as handle:
            handle.write(b"foreign")

        with self.assertRaisesRegex(ValueError, "does not match"):
            utils.require_matching_resume_files(
                self.working_directory_path, "md.tpr", "other.cpt")
        os.unlink(self.path("md.cpt"))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            utils.require_matching_resume_files(
                self.working_directory_path, "md.tpr", "md.cpt")

    def test_resume_handler_does_not_launch_with_a_foreign_checkpoint(self):
        self.write_resume_pair()
        with open(self.path("other.cpt"), "wb") as handle:
            handle.write(b"foreign")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(module, "subprocess") as subprocess_module:
                _, status, state, button = module.on_continue_prod_md(
                    self.working_directory_path, "md.tpr", "other.cpt",
                    1, 1, False, False, utils.ProcessStateDict())

            subprocess_module.Popen.assert_not_called()
            self.assertIn("does not match", self.plain_text(status))
            self.assertFalse(state["running"])
            self.assertEqual(button["value"], "Start")

    def test_checkpoint_dropdown_never_falls_back_to_another_run(self):
        import inspect

        with open(self.path("md.tpr"), "wb") as handle:
            handle.write(b"test")
        with open(self.path("other.cpt"), "wb") as handle:
            handle.write(b"foreign")

        expected_indexes = {
            protein_workflow: 25,
            complex_workflow: 30,
        }
        for module, checkpoint_index in expected_indexes.items():
            values = {
                name: "unused.txt"
                for name in inspect.signature(module.on_file_list_change).parameters
            }
            values["working_directory_path"] = self.working_directory_path
            values["prod_md_run_input_file_name"] = "md.tpr"
            with self.subTest(module=module.__name__):
                update = module.on_file_list_change(**values)[checkpoint_index]
                self.assertEqual(update["choices"], ["other.cpt"])
                self.assertIsNone(update["value"])

        with open(self.path("md.cpt"), "wb") as handle:
            handle.write(b"matching")
        for module, checkpoint_index in expected_indexes.items():
            values = {
                name: "unused.txt"
                for name in inspect.signature(module.on_file_list_change).parameters
            }
            values["working_directory_path"] = self.working_directory_path
            values["prod_md_run_input_file_name"] = "md.tpr"
            with self.subTest(module=module.__name__, matching=True):
                update = module.on_file_list_change(**values)[checkpoint_index]
                self.assertEqual(update["value"], "md.cpt")


class InteractiveGroupTests(WorkingDirectoryTestCase):
    def test_group_answers_are_resolved_by_name_not_fixed_number(self):
        command = ["gmx", "trjconv", "-f", "traj.xtc", "-o", self.path("out.xtc")]
        with unittest.mock.patch.object(
                utils, "probe_gmx_groups",
                return_value={"System": "7", "Protein": "12", "Backbone": "19"}) as probe:
            answers = utils.get_gmx_group_input(
                command, ["Backbone", "System"], self.working_directory_path)

        self.assertEqual(answers, "19\n7\n")
        probed_command = probe.call_args.args[0]
        self.assertNotEqual(probed_command[probed_command.index("-o") + 1], self.path("out.xtc"))
        self.assertEqual(probe.call_args.kwargs["cwd"], os.path.abspath(self.working_directory_path))

    def test_missing_named_group_has_an_actionable_error(self):
        with unittest.mock.patch.object(utils, "probe_gmx_groups", return_value={"System": "0"}):
            with self.assertRaisesRegex(Exception, "Backbone.*Available groups: System"):
                utils.get_gmx_group_input(
                    ["gmx", "trjconv", "-o", self.path("out.xtc")],
                    ["Backbone"], self.working_directory_path)


if __name__ == "__main__":
    unittest.main()
