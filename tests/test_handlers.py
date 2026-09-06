"""Tests for the Gradio callbacks that do not need GROMACS.

The callbacks are wrapped by path_security at import time, so these also exercise
that wrapper on the real signatures.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import unittest.mock

import MDAnalysis as mda
import pandas as pd

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
import path_security
import utils
from .testing_support import WorkingDirectoryTestCase, write_structure_pdb, write_trajectory

UNK_LIGAND_PDB = textwrap.dedent("""\
    HETATM    1  C1  UNK A 901      12.345  23.456  34.567  1.00  0.00           C
    HETATM    2  O1  UNK A 901      13.345  24.456  35.567  1.00  0.00           O
    END
    """)


class WorkingDirectoryCallbackTests(WorkingDirectoryTestCase):
    def test_opening_a_directory_creates_it_and_enables_the_actions(self):
        result = workflow.on_open_working_directory("_unittest_new_job")
        self.addCleanup(lambda: os.rmdir(os.path.join("data", "_unittest_new_job")))

        self.assertEqual(len(result), 5)
        dropdown, path, files, clean_button, upload = result
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(files, [])
        self.assertTrue(path.endswith("_unittest_new_job"))

    def test_directory_names_with_path_components_are_refused(self):
        """The callback warns and opens nothing rather than escaping ./data."""
        for value in ("../escape", "nested/job", "/absolute", "..\\escape"):
            with self.subTest(value=value):
                self.assertEqual(workflow.on_open_working_directory(value),
                                 (None, None, None, None, None))
        self.assertFalse(os.path.exists(os.path.join("data", "..", "escape")))

    def test_blank_directory_name_returns_nothing(self):
        self.assertEqual(workflow.on_open_working_directory("   "), (None, None, None, None, None))

    def test_existing_file_cannot_be_opened_as_a_job_directory(self):
        name = "_unittest_job_name_is_a_file"
        path = os.path.join("data", name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        for module, output_count in ((workflow, 5), (complex_workflow, 6)):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(module.gr, "Warning") as warning:
                self.assertEqual(
                    module.on_open_working_directory(name),
                    (None,) * output_count)
                warning.assert_called_once()

    def test_file_listing_hides_backups_and_zone_identifiers(self):
        for name in ("keep.gro", "#backup.gro.1#", "note.txt:Zone.Identifier"):
            with open(self.path(name), "w") as handle:
                handle.write("x")
        os.mkdir(self.path("subdir"))

        files = workflow.get_files_in_working_directory(self.working_directory_path)
        self.assertIn("keep.gro", files)
        self.assertNotIn("#backup.gro.1#", files)
        self.assertNotIn("subdir", files)
        self.assertFalse([name for name in files if name.endswith("Zone.Identifier")])

    def test_both_tabs_hide_exactly_the_same_files(self):
        """A job directory is browsable from either tab, so the two listings have
        to agree. They drifted once: only the complex tab hid the MM-PBSA scratch
        files, so the same job showed 109 files in one tab and 80 in the other.
        """
        for name in ("protein.gro", "topol.top", "FINAL_RESULTS_MMPBSA.dat",
                     "_GMXMMPBSA_COM.pdb", "_GMXMMPBSA_LIG.prmtop",
                     "#backup.gro.1#", "notes.txt:Zone.Identifier"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        listings = {module.__name__: module.get_files_in_working_directory(
            self.working_directory_path) for module in (workflow, complex_workflow)}

        self.assertEqual(*listings.values(), f"the two tabs disagree: {listings}")
        shown = next(iter(listings.values()))
        self.assertIn("FINAL_RESULTS_MMPBSA.dat", shown)
        self.assertFalse([name for name in shown if name.startswith("_GMXMMPBSA_")], shown)

    def test_missing_directory_lists_nothing(self):
        self.assertEqual(workflow.get_files_in_working_directory("data/_unittest_absent"), [])
        self.assertEqual(workflow.get_files_in_working_directory(None), [])

    def test_file_listing_is_sorted_by_name(self):
        """Every file dropdown is filtered out of this list, so its order is the UI's."""
        for name in ("zulu.gro", "alpha.top", "Bravo.mdp", "charlie.tpr"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                files = module.get_files_in_working_directory(self.working_directory_path)
                self.assertEqual(files, ["alpha.top", "Bravo.mdp", "charlie.tpr", "zulu.gro"])

    def test_working_directories_are_sorted_by_name(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                directories = module.get_working_directories()
                self.assertEqual(directories, sorted(directories, key=str.lower))
                self.assertIn(self.working_directory_name, directories)


class FileActionTests(WorkingDirectoryTestCase):
    def test_deleting_a_file_refreshes_the_listing(self):
        with open(self.path("doomed.gro"), "w") as handle:
            handle.write("x")
        files = workflow.on_delete_file(self.working_directory_path, "doomed.gro")
        self.assertNotIn("doomed.gro", files)

    def test_cleaning_removes_backups_only(self):
        with open(self.path("#topol.top.1#"), "w") as handle:
            handle.write("x")
        with open(self.path("topol.top"), "w") as handle:
            handle.write("x")

        workflow.on_clean_working_directory(self.working_directory_path)
        remaining = os.listdir(self.working_directory_path)
        self.assertIn("topol.top", remaining)
        self.assertNotIn("#topol.top.1#", remaining)

    def test_text_file_round_trip(self):
        with open(self.path("notes.mdp"), "w") as handle:
            handle.write("integrator = steep\n")

        viewer, save_button = workflow.on_view_text_file(self.working_directory_path, "notes.mdp")
        self.assertIn("integrator = steep", viewer["value"])

        workflow.on_save_text_file(self.working_directory_path, "notes.mdp", "integrator = md\n")
        with open(self.path("notes.mdp")) as handle:
            self.assertEqual(handle.read(), "integrator = md\n")

    def test_text_editor_rejects_binary_and_oversized_job_files(self):
        with open(self.path("trajectory.xtc"), "wb") as handle:
            handle.write(b"binary trajectory")
        with open(self.path("huge.log"), "wb") as handle:
            handle.truncate(path_security.MAX_EDITABLE_TEXT_BYTES + 1)

        for module in (workflow, complex_workflow):
            for name, expected in (("trajectory.xtc", "not a supported"),
                                   ("huge.log", "too large")):
                with self.subTest(module=module.__name__, name=name), \
                        unittest.mock.patch.object(module.gr, "Warning") as warning:
                    viewer, save = module.on_view_text_file(
                        self.working_directory_path, name)
                self.assertIsNone(viewer)
                self.assertIsNone(save)
                self.assertIn(expected, str(warning.call_args.args[0]))

            with open(self.path("trajectory.xtc"), "rb") as handle:
                original = handle.read()
            with unittest.mock.patch.object(module.gr, "Warning") as warning:
                module.on_save_text_file(
                    self.working_directory_path, "trajectory.xtc", "corrupt")
            with open(self.path("trajectory.xtc"), "rb") as handle:
                self.assertEqual(handle.read(), original)
            self.assertIn("not a supported", str(warning.call_args.args[0]))

    def test_export_writes_a_csv_into_the_job_directory(self):
        frame = pd.DataFrame({"Time (ns)": [0.0, 0.1], "RMSD": [0.0, 1.5]})
        files, status = workflow.on_export_df(self.working_directory_path, frame, "rmsd.csv")
        self.assertIn("rmsd.csv", files)
        self.assertIn("File exported", self.plain_text(status))
        self.assertTrue(os.path.exists(self.path("rmsd.csv")))

    def test_output_role_contracts_prevent_cross_type_overwrites_and_aliases(self):
        with open(self.path("valuable.tpr"), "wb") as handle:
            handle.write(b"known-good binary run input")
        with open(self.path("input.gro"), "w") as handle:
            handle.write("known-good structure")

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__, case="mdp-over-tpr"), \
                    self.assertRaisesRegex(ValueError, r"expected \.mdp"):
                module.on_generate_energy_minimization_mdp_file(
                    self.working_directory_path, "valuable.tpr",
                    "AMBER99SB-ILDN")
            with open(self.path("valuable.tpr"), "rb") as handle:
                self.assertEqual(handle.read(), b"known-good binary run input")

            with self.subTest(module=module.__name__, case="box-alias"), \
                    unittest.mock.patch.object(module, "run_checked_command") as run, \
                    self.assertRaisesRegex(ValueError, "cannot share"):
                module.on_generate_simulation_box(
                    self.working_directory_path, "input.gro", "input.gro",
                    "dodecahedron", 1.0, "AMBER99SB-ILDN")
            run.assert_not_called()
            with open(self.path("input.gro")) as handle:
                self.assertEqual(handle.read(), "known-good structure")

            with self.subTest(module=module.__name__, case="export-extension"), \
                    self.assertRaisesRegex(ValueError, r"expected \.csv"):
                module.on_export_df(
                    self.working_directory_path,
                    pd.DataFrame({"x": [1]}), "valuable.tpr")

    def test_upload_destination_must_remain_a_pdb(self):
        descriptor, source = tempfile.mkstemp(suffix=".pdb")
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(source) and os.remove(source))
        with open(source, "w") as handle:
            handle.write(UNK_LIGAND_PDB)

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    self.assertRaisesRegex(ValueError, r"expected \.pdb"):
                module.on_upload_protein_structure_file(
                    self.working_directory_path, "valuable.tpr", source)


class AccordionAndSelectionTests(unittest.TestCase):
    def test_selecting_a_structure_opens_its_accordion(self):
        button, accordion = workflow.on_selected_structure_file_state_change("protein.gro")
        self.assertTrue(button["interactive"])
        self.assertTrue(accordion["open"])

    def test_deselecting_disables_the_button_without_closing_the_accordion(self):
        """Closing it would collapse a viewer the user is reading."""
        button, accordion = workflow.on_selected_structure_file_state_change(None)
        self.assertFalse(button["interactive"])
        self.assertNotIn("open", accordion)

    def test_text_selection_behaves_the_same_way(self):
        button, accordion = workflow.on_selected_text_file_state_change("topol.top")
        self.assertTrue(button["interactive"])
        self.assertTrue(accordion["open"])
        button, accordion = workflow.on_selected_text_file_state_change(None)
        self.assertFalse(button["interactive"])
        self.assertNotIn("open", accordion)

    def test_ion_method_switch_shows_the_matching_inputs(self):
        concentration, cation_charge, anion_charge, *counts = \
            workflow.on_add_ions_method_change("Concentration")
        self.assertTrue(concentration["visible"])
        self.assertTrue(cation_charge["visible"])
        self.assertTrue(anion_charge["visible"])
        self.assertTrue(all(not update["visible"] for update in counts))

        concentration, cation_charge, anion_charge, *counts = \
            workflow.on_add_ions_method_change("Number")
        self.assertFalse(concentration["visible"])
        self.assertTrue(cation_charge["visible"])
        self.assertTrue(anion_charge["visible"])
        self.assertTrue(all(update["visible"] for update in counts))

    def test_mdp_type_switch_renames_the_parameter_file(self):
        seed_box, file_name = workflow.on_change_mdp_type("Initial")
        # Production inherits equilibrated velocities, so neither mode asks for
        # a new random seed.
        self.assertFalse(seed_box["visible"])
        self.assertEqual(file_name, "md_initial.mdp")

        seed_box, file_name = workflow.on_change_mdp_type("Continuation")
        self.assertFalse(seed_box["visible"])
        self.assertEqual(file_name, "md_continue.mdp")

    def test_button_state_follows_the_process_state(self):
        state = utils.ProcessStateDict()
        self.assertEqual(workflow.sync_button_state(state)["value"], "Start")
        state["running"] = True
        self.assertEqual(workflow.sync_button_state(state)["value"], "Stop")


class AsyncProcessHandlerTests(WorkingDirectoryTestCase):
    class FakeProcess:
        def __init__(self, returncode=None):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def test_timer_publishes_completion_status_and_refreshes_files(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                output_name = f"{module.__name__}.gro"
                with open(self.path(output_name), "w") as handle:
                    handle.write("finished")
                state = utils.ProcessStateDict()
                proc = self.FakeProcess(0)
                key = utils.get_process_job_key(self.working_directory_path, "nvt")
                utils.set_process_running(state, proc, key, "NVT equilibration",
                                          self.working_directory_path)

                files, status, button = module.sync_process_state(
                    self.working_directory_path, state)

                self.assertIn(output_name, files)
                self.assertIn("NVT equilibration completed successfully", status)
                self.assertEqual(button["value"], "Start")
                self.assertEqual(state["returncode"], 0)

    def test_initial_and_continuation_cannot_write_the_same_prefix_together(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                for name in ("md.tpr", "md.cpt"):
                    with open(self.path(name), "wb") as handle:
                        handle.write(b"test")
                initial_state = utils.ProcessStateDict()
                continuation_state = utils.ProcessStateDict()
                proc = self.FakeProcess(None)
                key = utils.get_process_job_key(self.working_directory_path, "md")
                self.addCleanup(utils.release_process_job, key, proc)

                with unittest.mock.patch.object(module, "subprocess") as fake_subprocess, \
                        unittest.mock.patch.object(module, "threading"):
                    fake_subprocess.Popen.return_value = proc
                    module.on_run_prod_md(self.working_directory_path, "md.tpr", 1, 1,
                                          False, False, initial_state)
                    _, status, returned_state, button = module.on_continue_prod_md(
                        self.working_directory_path, "md.tpr", "md.cpt", 1, 1,
                        False, False, continuation_state)

                fake_subprocess.Popen.assert_called_once()
                self.assertIs(returned_state["proc"], proc)
                self.assertTrue(returned_state["running"])
                self.assertEqual(button["value"], "Stop")
                self.assertIn("already running", self.plain_text(status))
                utils.clear_process_state(initial_state)
                utils.clear_process_state(continuation_state)
                utils.release_process_job(key, proc)


class MdpCallbackTests(WorkingDirectoryTestCase):
    def test_each_generator_writes_its_file_and_lists_it(self):
        cases = (
            (workflow.on_generate_ions_mdp_file, ("ions.mdp", "CHARMM36"), "ions.mdp"),
            (workflow.on_generate_energy_minimization_mdp_file, ("em.mdp", "CHARMM36"), "em.mdp"),
        )
        for callback, extra, expected in cases:
            with self.subTest(callback=callback.__name__):
                files, status = callback(self.working_directory_path, *extra)
                self.assertIn(expected, files)
                self.assertIn("successfully", self.plain_text(status))
                with open(self.path(expected)) as handle:
                    self.assertIn("force-switch", handle.read())

    def test_nvt_mdp_callback_passes_the_temperature_through(self):
        files, _ = workflow.on_generate_nvt_equilibration_mdp_file(
            self.working_directory_path, 100, 0.002, 305, "nvt.mdp", "AMBER99SB-ILDN")
        self.assertIn("nvt.mdp", files)
        with open(self.path("nvt.mdp")) as handle:
            content = handle.read()
        self.assertIn("ref_t       = 305", content)
        self.assertIn("gen_temp    = 305", content)
        self.assertIn("-DPOSRES", content)

    def test_mdp_generators_reject_timesteps_that_need_hmr(self):
        cases = (
            (workflow, "on_generate_nvt_equilibration_mdp_file",
             (100, 0.003, 300, "unsafe_nvt.mdp", "AMBER99SB-ILDN")),
            (workflow, "on_generate_npt_equilibration_mdp_file",
             (100, 0.003, 300, 1.0, "unsafe_npt.mdp", "AMBER99SB-ILDN")),
            (workflow, "on_generate_prod_md_mdp_file",
             (1, 0.003, 300, 1.0, "Initial", 0, "unsafe_md.mdp", False,
              "ani2x", "Protein", "AMBER99SB-ILDN")),
            (complex_workflow, "on_generate_nvt_equilibration_mdp_file",
             (100, 0.003, 300, "unsafe_complex_nvt.mdp", "AMBER99SB-ILDN")),
            (complex_workflow, "on_generate_npt_equilibration_mdp_file",
             (100, 0.003, 300, 1.0, "unsafe_complex_npt.mdp", "AMBER99SB-ILDN")),
            (complex_workflow, "on_generate_prod_md_mdp_file",
             (1, 0.003, 300, 1.0, "Initial", 0, "unsafe_complex_md.mdp",
              False, "ani2x", "Protein_LIG", "AMBER99SB-ILDN")),
        )
        for module, handler_name, arguments in cases:
            with self.subTest(module=module.__name__, handler=handler_name):
                files, status = getattr(module, handler_name)(
                    self.working_directory_path, *arguments)
                self.assertIn("0.002 ps", self.plain_text(status))
                self.assertIn("HMR", self.plain_text(status))
                expected = next(arg for arg in arguments
                                if isinstance(arg, str) and arg.startswith("unsafe"))
                self.assertNotIn(expected, files)


class WorkflowSafetyContractTests(WorkingDirectoryTestCase):
    GROMPP_HANDLERS = (
        "on_generate_ions_tpr_file",
        "on_generate_energy_minimization_tpr_file",
        "on_generate_nvt_equilibration_tpr_file",
        "on_generate_npt_equilibration_tpr_file",
        "on_generate_prod_md_tpr_file",
    )

    @staticmethod
    def tpr_arguments(max_warnings=0, force_field=None):
        arguments = ["input.gro", "topol.top", "step.mdp", "step.tpr",
                     max_warnings]
        if force_field is not None:
            arguments.append(force_field)
        return tuple(arguments)

    def test_nonzero_maxwarn_is_never_reported_as_green_success(self):
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "amber99sb-ildn.ff/forcefield.itp"\n')
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "rlist = 1.0\nrvdw = 1.0\nrcoulomb = 1.0\n"
                "DispCorr = EnerPres\ncoulombtype = PME\n"
                "cutoff-scheme = Verlet\n")
        for module in (workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(module, "run_checked_command"):
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path,
                        *self.tpr_arguments(max_warnings=2))
                self.assertIn("color:orange", status)
                self.assertIn("Expert override", self.plain_text(status))
                self.assertIn("2 warning", self.plain_text(status))

    def _write_gromos_grompp_inputs(self):
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "gromos54a7.ff/forcefield.itp"\n')
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "integrator = steep\ncutoff-scheme = Verlet\n"
                "rlist = 1.4\nrvdw = 1.4\nrcoulomb = 1.4\n"
                "coulombtype = PME\nDispCorr = no\n")

    def test_only_the_known_gromos_warning_gets_a_scoped_allowance(self):
        self._write_gromos_grompp_inputs()
        warning = """WARNING 1 [file topol.top, line 2]:
  The GROMOS force fields have been parametrized with a physically incorrect
  multiple-time-stepping scheme for a twin-range cut-off. When used with a
  single-range cut-off, physical properties might differ.

There was 1 WARNING
"""
        for module in (workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                commands = []

                def succeed(command, cwd):
                    commands.append(command)
                    with open(command[command.index("-o") + 1], "w") as handle:
                        handle.write("new tpr")
                    with open(command[command.index("-po") + 1], "w") as handle:
                        handle.write("processed mdp")
                    return subprocess.CompletedProcess(
                        command, 0, stdout="", stderr=warning)

                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(
                            module, "run_checked_command", side_effect=succeed):
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path,
                        *self.tpr_arguments(force_field="GROMOS54A7"))

                self.assertIn("color:orange", status)
                self.assertIn("historical twin-range", self.plain_text(status))
                self.assertEqual(
                    commands[0][commands[0].index("-maxwarn") + 1], "1")
                self.assertIn(".grompp_stage_", commands[0][commands[0].index("-o") + 1])
                with open(self.path("step.tpr")) as handle:
                    self.assertEqual(handle.read(), "new tpr")

    def test_an_unrelated_gromos_warning_preserves_the_previous_tpr(self):
        self._write_gromos_grompp_inputs()
        with open(self.path("step.tpr"), "w") as handle:
            handle.write("known-good tpr")

        unrelated = """WARNING 1 [file topol.top, line 7]:
  Atom names do not match and this is not the reviewed warning.

There was 1 WARNING
"""

        def succeed_with_wrong_warning(command, cwd):
            with open(command[command.index("-o") + 1], "w") as handle:
                handle.write("unsafe replacement")
            with open(command[command.index("-po") + 1], "w") as handle:
                handle.write("processed mdp")
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=unrelated)

        with unittest.mock.patch.object(
                workflow, "run_checked_command",
                side_effect=succeed_with_wrong_warning):
            _, status = workflow.on_generate_energy_minimization_tpr_file(
                self.working_directory_path, "input.gro", "topol.top",
                "step.mdp", "step.tpr", 0, "GROMOS54A7")

        self.assertIn("not covered", self.plain_text(status))
        with open(self.path("step.tpr")) as handle:
            self.assertEqual(handle.read(), "known-good tpr")

    def test_maxwarn_never_silently_truncates_or_accepts_nonfinite_values(self):
        invalid_values = (True, -1, 1.5, 11, float("nan"), float("inf"), "2")
        for module in (workflow, complex_workflow):
            for value in invalid_values:
                with self.subTest(module=module.__name__, value=value), \
                        self.assertRaisesRegex(ValueError, "integer from 0 to 10"):
                    module._normalise_max_warnings(value)
            self.assertEqual(module._normalise_max_warnings(0), 0)
            self.assertEqual(module._normalise_max_warnings(10.0), 10)

    def test_nonfinite_box_padding_is_rejected_before_gromacs_runs(self):
        for module in (workflow, complex_workflow):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(module=module.__name__, value=value), \
                        unittest.mock.patch.object(module, "run_checked_command") as run:
                    _, status = module.on_generate_simulation_box(
                        self.working_directory_path, "input.gro", "box.gro",
                        "dodecahedron", value, "AMBER99SB-ILDN")
                run.assert_not_called()
                self.assertIn("finite number", self.plain_text(status))

                with self.assertRaisesRegex(ValueError, "finite number"):
                    module.on_force_field_change_for_box("AMBER99SB-ILDN", value)

    def test_every_tpr_handler_blocks_an_mdp_incompatible_with_topology(self):
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "charmm36.ff/forcefield.itp"\n')
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "rlist = 1.0\nrvdw = 1.0\nrcoulomb = 1.0\n"
                "DispCorr = EnerPres\n")
        for module in (workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(module, "run_checked_command") as run:
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path,
                        *self.tpr_arguments(force_field=None))
                run.assert_not_called()
                plain_status = self.plain_text(status)
                self.assertIn("MDP 'step.mdp' is incompatible", plain_status)
                self.assertIn("CHARMM", plain_status)

    def test_custom_force_field_success_carries_an_expert_warning(self):
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "my-lab-force-field.ff/forcefield.itp"\n')
        with open(self.path("step.mdp"), "w") as handle:
            handle.write(
                "integrator = md\ndt = 0.002\nrlist = 0.9\nrvdw = 0.9\n"
                "rcoulomb = 0.9\ncutoff-scheme = Verlet\n")
        for module in (workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(module, "run_checked_command"):
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path,
                        *self.tpr_arguments(force_field=None))
                self.assertIn("color:orange", status)
                plain_status = self.plain_text(status)
                self.assertIn("Custom force field 'my-lab-force-field'", plain_status)
                self.assertIn("you are responsible", plain_status)

    def test_every_tpr_handler_blocks_a_topology_from_another_force_family(self):
        with open(self.path("topol.top"), "w") as handle:
            handle.write('#include "charmm36.ff/forcefield.itp"\n')
        for module in (workflow, complex_workflow):
            for handler_name in self.GROMPP_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(module, "run_checked_command") as run:
                    _, status = getattr(module, handler_name)(
                        self.working_directory_path,
                        *self.tpr_arguments(force_field="AMBER99SB-ILDN"))
                run.assert_not_called()
                self.assertIn("does not match", self.plain_text(status))

    def test_box_padding_floor_depends_on_force_field_family(self):
        cases = (("GROMOS54A7", 1.3, "1.4"),
                 ("CHARMM36", 1.1, "1.2"),
                 ("AMBER99SB-ILDN", 0.9, "1.0"))
        for module in (workflow, complex_workflow):
            for force_field, distance, expected in cases:
                with self.subTest(module=module.__name__, force_field=force_field), \
                        unittest.mock.patch.object(module, "run_checked_command") as run:
                    _, status = module.on_generate_simulation_box(
                        self.working_directory_path, "input.gro", "box.gro",
                        "dodecahedron", distance, force_field)
                run.assert_not_called()
                self.assertIn(expected + " nm", self.plain_text(status))

    def test_water_site_mapping_and_existing_topology_are_both_validated(self):
        expected = {
            "TIP3P": "spc216.gro", "SPCE": "spc216.gro",
            "OPC3": "spc216.gro", "TIP4P": "tip4p.gro",
            "TIP4P-Ew": "tip4p.gro", "OPC": "tip4p.gro",
            "TIP5P": "tip5p.gro",
        }
        for module in (workflow, complex_workflow):
            for model, coordinates in expected.items():
                with self.subTest(module=module.__name__, model=model):
                    self.assertEqual(
                        module._solvent_configuration_for_water_model(model),
                        coordinates)

            with open(self.path("old.top"), "w") as handle:
                handle.write('#include "amber99sb-ildn.ff/tip4p.itp"\n')
            with unittest.mock.patch.object(module, "run_checked_command") as run:
                _, status = module.on_solvate_protein(
                    self.working_directory_path, "box.gro", "solvated.gro",
                    "old.top", "solvated.top", "spc216.gro", "TIP3P")
            run.assert_not_called()
            self.assertIn("contains TIP4P", self.plain_text(status))

    def test_force_field_change_filters_water_models_and_updates_geometry(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                box, water, solvent = module.on_force_field_change(
                    "GROMOS54A7", 1.0, "TIP3P")
                self.assertEqual(box["minimum"], 1.4)
                self.assertEqual(box["value"], 1.4)
                self.assertEqual(water["choices"], ["NONE", "SPC", "SPCE"])
                self.assertEqual(water["value"], "SPC")
                self.assertEqual(solvent["value"], "spc216.gro")

                _, amber19_water, amber19_solvent = module.on_force_field_change(
                    "AMBER19SB", 1.0, "TIP3P")
                self.assertEqual(amber19_water["value"], "OPC")
                self.assertEqual(amber19_solvent["value"], "tip4p.gro")

                _, opls_water, opls_solvent = module.on_force_field_change(
                    "OPLSAA", 1.0, "TIP3P")
                self.assertEqual(opls_water["value"], "TIP4P")
                self.assertEqual(opls_solvent["value"], "tip4p.gro")

                _, preserved_water, _ = module.on_force_field_change(
                    "OPLSAA", 1.0, "SPCE")
                self.assertEqual(preserved_water["value"], "SPCE")

                _, custom_water, _ = module.on_force_field_change(
                    "my-custom-ff", 1.0, "OPC")
                custom_values = [module._water_choice_value(choice)
                                 for choice in custom_water["choices"]]
                self.assertIn("OPC", custom_values)
                self.assertIn("TIPS3P", custom_values)

    def test_topology_generation_rejects_unsupported_bundled_water_model(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(module, "run_checked_command") as run:
                _, status = module.on_generate_protein_topology(
                    self.working_directory_path, "protein.pdb", "protein.gro",
                    "topol.top", "GROMOS54A7", "TIP3P",
                    module.DEFAULT_TERMINUS_CHOICE,
                    module.DEFAULT_TERMINUS_CHOICE)
            run.assert_not_called()
            plain_status = self.plain_text(status)
            self.assertIn("not supported", plain_status)
            self.assertIn("GROMOS54A7", plain_status)

    def test_complex_merge_dropdowns_keep_gro_top_and_itp_contracts_separate(self):
        for name in ("protein.pdb", "protein.gro", "ligand.gro",
                     "system.top", "ligand.itp"):
            with open(self.path(name), "w") as handle:
                handle.write("x")
        import inspect
        values = {name: "unused.txt" for name in inspect.signature(
            complex_workflow.on_file_list_change).parameters}
        values["working_directory_path"] = self.working_directory_path
        updates = complex_workflow.on_file_list_change(**values)

        self.assertEqual(sorted(updates[3]["choices"]), ["ligand.gro", "protein.gro"])
        self.assertEqual(sorted(updates[4]["choices"]), ["ligand.gro", "protein.gro"])
        self.assertEqual(updates[5]["choices"], ["system.top"])
        self.assertEqual(updates[6]["choices"], ["ligand.itp"])

    def test_acpype_gaff_is_blocked_with_non_amber_protein_force_fields(self):
        with unittest.mock.patch.object(
                complex_workflow, "run_checked_command") as run:
            _, status = complex_workflow.on_generate_ligand_topology(
                self.working_directory_path, "ligand.pdb", "ligand", 0,
                "bcc", "gaff2", "CHARMM36")
        run.assert_not_called()
        self.assertIn("AMBER-family", self.plain_text(status))
        self.assertIn("CHARMM36", self.plain_text(status))

    def test_merge_blocks_gaff_but_allows_an_unrelated_ligand_itp(self):
        with open(self.path("protein.top"), "w") as handle:
            handle.write('#include "charmm36.ff/forcefield.itp"\n')
        with open(self.path("ligand.itp"), "w") as handle:
            handle.write("; generated by CGenFF\n[ moleculetype ]\nLIG 3\n")

        with unittest.mock.patch.object(
                complex_workflow, "merge_protein_ligand_topologies") as merge:
            _, status = complex_workflow.on_merge_topologies(
                self.working_directory_path, "protein.top", "ligand.itp",
                "complex.top", "CHARMM36")
        merge.assert_called_once()
        self.assertIn("successfully", self.plain_text(status))

        with open(self.path("ligand.itp"), "w") as handle:
            handle.write("; generated by ACPYPE using GAFF2\n[ moleculetype ]\nLIG 3\n")
        with unittest.mock.patch.object(
                complex_workflow, "merge_protein_ligand_topologies") as merge:
            _, status = complex_workflow.on_merge_topologies(
                self.working_directory_path, "protein.top", "ligand.itp",
                "complex.top", "CHARMM36")
        merge.assert_not_called()
        self.assertIn("AMBER-family", self.plain_text(status))

    def test_merge_handlers_reject_wrong_file_contracts(self):
        with unittest.mock.patch.object(
                complex_workflow, "merge_protein_ligand_structures") as merge:
            with self.assertRaisesRegex(ValueError, r"expected \.gro"):
                complex_workflow.on_merge_structures(
                    self.working_directory_path, "protein.pdb", "ligand.gro",
                    "complex.gro")
        merge.assert_not_called()

        with unittest.mock.patch.object(
                complex_workflow, "merge_protein_ligand_topologies") as merge:
            with self.assertRaisesRegex(ValueError, r"expected \.top"):
                complex_workflow.on_merge_topologies(
                    self.working_directory_path, "protein.itp", "ligand.top",
                    "complex.top", "AMBER99SB-ILDN")
        merge.assert_not_called()

    def test_merge_handlers_validate_the_selected_ligand_pair(self):
        protein_gro = textwrap.dedent("""\
            Protein
                1
                1ALA     CA    1   1.000   1.000   1.000
               5.00000   5.00000   5.00000
            """)
        ligand_gro = textwrap.dedent("""\
            Ligand
                1
                1LIG     C1    1   1.300   1.300   1.300
               1.00000   1.00000   1.00000
            """)
        ligand_itp = textwrap.dedent("""\
            [ moleculetype ]
            ligand 3
            [ atoms ]
            1 c3 1 LIG C1 1 0.0 12.011
            """)
        for name, content in (
                ("protein.gro", protein_gro),
                ("first_GMX.gro", ligand_gro),
                ("second_GMX.itp", ligand_itp)):
            with open(self.path(name), "w") as handle:
                handle.write(content)

        files, status = complex_workflow.on_merge_structures(
            self.working_directory_path, "protein.gro", "first_GMX.gro",
            "complex.gro", "second_GMX.itp")

        self.assertNotIn("complex.gro", files)
        self.assertIn("different ACPYPE output sets", self.plain_text(status))

    def test_far_ligand_merge_returns_an_orange_coordinate_frame_warning(self):
        files_to_write = {
            "protein.gro": textwrap.dedent("""\
                Protein
                    1
                    1ALA     CA    1   1.000   1.000   1.000
                   5.00000   5.00000   5.00000
                """),
            "ligand_GMX.gro": textwrap.dedent("""\
                Ligand
                    1
                    1LIG     C1    1  20.000  20.000  20.000
                   1.00000   1.00000   1.00000
                """),
            "ligand_GMX.itp": textwrap.dedent("""\
                [ moleculetype ]
                ligand 3
                [ atoms ]
                1 c3 1 LIG C1 1 0.0 12.011
                """),
        }
        for name, content in files_to_write.items():
            with open(self.path(name), "w") as handle:
                handle.write(content)

        files, status = complex_workflow.on_merge_structures(
            self.working_directory_path, "protein.gro", "ligand_GMX.gro",
            "complex.gro", "ligand_GMX.itp")

        self.assertIn("complex.gro", files)
        self.assertIn("color:orange", status)
        self.assertIn("same coordinate frame", self.plain_text(status))


class ShippedSafetyDefaultsTests(unittest.TestCase):
    def test_timestep_and_maxwarn_components_ship_with_safe_defaults(self):
        import gradio as gr
        import webui

        blocks = list(webui.blocks.blocks.values())
        time_steps = [block for block in blocks if isinstance(block, gr.Slider)
                      and str(getattr(block, "label", "")).startswith("Time Step")]
        self.assertEqual(len(time_steps), 6)
        for slider in time_steps:
            self.assertLessEqual(slider.maximum, 0.002)
            self.assertLessEqual(slider.value, 0.002)
            self.assertIn("no HMR", slider.label)

        maxwarn = [block for block in blocks if isinstance(block, gr.Slider)
                   and str(getattr(block, "label", "")).startswith("Max Warnings")]
        self.assertEqual(len(maxwarn), 2)
        for slider in maxwarn:
            self.assertEqual(slider.value, 0)
            self.assertIn("dangerous", slider.label)

    def test_force_field_is_wired_into_every_tpr_and_box_handler(self):
        import webui

        expected = {
            "on_generate_ions_tpr_file",
            "on_generate_energy_minimization_tpr_file",
            "on_generate_nvt_equilibration_tpr_file",
            "on_generate_npt_equilibration_tpr_file",
            "on_generate_prod_md_tpr_file",
            "on_generate_simulation_box",
        }
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        found = []
        for handler in handlers:
            if getattr(handler.fn, "__name__", "") not in expected:
                continue
            found.append((handler.fn.__module__, handler.fn.__name__))
            self.assertIn(getattr(handler.inputs[-1], "label", None),
                          ("Force Field",))
        self.assertEqual(len(found), len(expected) * 2)

    def test_ion_charge_controls_are_visible_in_default_concentration_mode(self):
        import gradio as gr
        import webui

        charge_sliders = [
            block for block in webui.blocks.blocks.values()
            if isinstance(block, gr.Slider)
            and getattr(block, "label", None) in ("Cation Charge", "Anion Charge")
        ]
        self.assertEqual(len(charge_sliders), 4)
        self.assertTrue(all(slider.visible for slider in charge_sliders))


class TrajectoryViewerCallbackTests(WorkingDirectoryTestCase):
    def test_viewing_a_trajectory_returns_an_iframe_and_a_species_legend(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=3, ions={"NA": 2})
        write_trajectory(structure, self.path("traj.xtc"), n_frames=6)

        html, status = workflow.on_view_trajectory(self.working_directory_path, "system.pdb",
                                                   "traj.xtc", "Protein + Ligand + Ions", 200)
        self.assertIn("<iframe", html)
        match = re.search(r'/static/([^"?]+)_view\.html', html)
        self.assertIsNotNone(match)
        static_basename = match.group(1)
        text = self.plain_text(status)
        self.assertIn("Showing 6 of 6 frames", text)
        self.assertIn("NA 2", text)
        self.assertTrue(os.path.exists(utils.STATIC_ROOT / f"{static_basename}_view.html"))

    def test_missing_selection_warns_instead_of_raising(self):
        self.assertEqual(workflow.on_view_trajectory(self.working_directory_path, None, None,
                                                     "Protein", 200), (None, None))

    def test_unreadable_pair_reports_an_error_status(self):
        write_structure_pdb(self.path("system.pdb"), n_residues=2)
        with open(self.path("broken.xtc"), "wb") as handle:
            handle.write(b"not really an xtc")

        html, status = workflow.on_view_trajectory(self.working_directory_path, "system.pdb",
                                                   "broken.xtc", "Protein", 200)
        self.assertIsNone(html)
        self.assertIn("Error", self.plain_text(status))

    def test_the_two_tabs_write_to_different_static_files(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=3)
        write_trajectory(structure, self.path("traj.xtc"), n_frames=3)

        protein_html, _ = workflow.on_view_trajectory(
            self.working_directory_path, "system.pdb", "traj.xtc", "Protein", 200)
        complex_html, _ = complex_workflow.on_view_trajectory(
            self.working_directory_path, "system.pdb", "traj.xtc", "Protein", 200)
        protein_basename = re.search(
            r'/static/([^"?]+)_view\.html', protein_html).group(1)
        complex_basename = re.search(
            r'/static/([^"?]+)_view\.html', complex_html).group(1)
        self.assertNotEqual(protein_basename, complex_basename)
        self.assertTrue(os.path.exists(utils.STATIC_ROOT / f"{protein_basename}.xtc"))
        self.assertTrue(os.path.exists(utils.STATIC_ROOT / f"{complex_basename}.xtc"))


class FileListChangeTests(WorkingDirectoryTestCase):
    def arguments(self, count):
        # path_security rejects an empty file name, so use a harmless placeholder
        return ["unused.txt"] * count

    def test_dropdowns_point_at_the_files_each_step_produces(self):
        for name in ("protein.pdb", "protein.gro", "topology.top", "em.mdp", "em.tpr", "md.xtc"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        import inspect
        parameters = inspect.signature(workflow.on_file_list_change).parameters
        values = {name: "unused.txt" for name in parameters}
        values["working_directory_path"] = self.working_directory_path
        values["protein_structure_file_name"] = "protein.pdb"
        values["topology_output_file_name"] = "protein.gro"
        values["topology_output_topology_file_name"] = "topology.top"

        result = workflow.on_file_list_change(**values)

        table = result[0]
        self.assertEqual(sorted(table["File"]), sorted(os.listdir(self.working_directory_path)))
        # first dropdown is the topology input, which should follow the uploaded pdb
        self.assertEqual(result[1]["value"], "protein.pdb")
        # the box input follows pdb2gmx's output structure
        self.assertEqual(result[2]["value"], "protein.gro")

    def test_the_results_dropdown_offers_only_the_canonical_mmpbsa_summary(self):
        """Arbitrary .dat files have no guaranteed companion CSV/decomposition."""
        for name in ("FINAL_RESULTS_MMPBSA.dat", "older_run.dat", "notes.txt"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        import inspect
        parameters = inspect.signature(complex_workflow.on_file_list_change).parameters
        values = {name: "unused.txt" for name in parameters}
        values["working_directory_path"] = self.working_directory_path

        results = complex_workflow.on_file_list_change(**values)[-1]
        self.assertEqual(results["choices"], ["FINAL_RESULTS_MMPBSA.dat"])
        self.assertEqual(results["value"], "FINAL_RESULTS_MMPBSA.dat")

    def test_a_canonical_legacy_summary_remains_loadable_from_the_dropdown(self):
        legacy = self.path("mmpbsa")
        os.mkdir(legacy)
        with open(os.path.join(legacy, "FINAL_RESULTS_MMPBSA.dat"), "w") as handle:
            handle.write("legacy")
        with open(self.path("unrelated.dat"), "w") as handle:
            handle.write("other")

        import inspect
        parameters = inspect.signature(complex_workflow.on_file_list_change).parameters
        values = {name: "unused.txt" for name in parameters}
        values["working_directory_path"] = self.working_directory_path
        results = complex_workflow.on_file_list_change(**values)[-1]

        self.assertEqual(results["choices"], ["FINAL_RESULTS_MMPBSA.dat"])
        self.assertEqual(results["value"], "FINAL_RESULTS_MMPBSA.dat")

    def test_every_dropdown_is_offered_files_of_its_own_kind(self):
        """Order, not just count.

        The return tuple and the .change() outputs list are matched positionally,
        so inserting an update in the wrong place silently feeds every later
        dropdown its neighbour's files. The count test cannot see that: the
        lengths still agree. This checks each component against what its label
        says it wants.
        """
        import gradio as gr
        import webui

        expected = {
            "Structure File Name": (".pdb", ".gro"),
            "Input Topology File Name": (".top",),
            "Parameter File Name": (".mdp",),
            "Run Input File Name": (".tpr",),
            "Run Input File Name (.tpr)": (".tpr",),
            "Checkpoint File Name": (".cpt",),
            "Input Trajectory File Name": (".xtc", ".trr"),
            "Protein Structure File Name": (".gro",),
            "Ligand Structure File Name": (".gro",),
            "Protein Topology File Name": (".top",),
            "Ligand Topology File Name": (".itp",),
            "Results File Name": (".dat",),
        }
        for name in ("protein.pdb", "protein.gro", "topol.top", "ligand.itp", "em.mdp",
                     "md.tpr", "md.cpt", "md.xtc", "md.trr", "FINAL_RESULTS_MMPBSA.dat"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        checked = 0
        for handler in handlers:
            if getattr(handler.fn, "__name__", "") != "on_file_list_change":
                continue
            import inspect
            required = len(inspect.signature(handler.fn).parameters)
            returned = handler.fn(self.working_directory_path, *self.arguments(required - 1))

            for component, update in zip(handler.outputs, returned):
                if not isinstance(component, gr.Dropdown):
                    continue
                suffixes = expected.get(component.label)
                if suffixes is None:
                    continue
                for choice in update["choices"]:
                    name = choice[1] if isinstance(choice, (list, tuple)) else choice
                    with self.subTest(label=component.label, choice=name):
                        self.assertTrue(name.endswith(suffixes),
                                        f"{component.label!r} was offered {name!r}")
                    checked += 1

        self.assertGreater(checked, 40, "expected to check many dropdown choices")

    def test_return_count_matches_the_wired_outputs(self):
        """A mismatch here breaks every file refresh at runtime."""
        import webui
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        for handler in handlers:
            if getattr(handler.fn, "__name__", "") != "on_file_list_change":
                continue
            import inspect
            required = len(inspect.signature(handler.fn).parameters)
            returned = handler.fn(self.working_directory_path, *self.arguments(required - 1))
            self.assertIsInstance(returned, tuple)
            self.assertEqual(len(returned), len(handler.outputs))


class WiringTests(unittest.TestCase):
    """Every handler must return as many values as it is wired to display.

    FileListChangeTests guards on_file_list_change by calling it; this covers the
    rest statically, including the ones that would need a real GROMACS run or a
    launched subprocess to reach. A handler wired to one output too many fails at
    runtime with nothing in the logs to explain it.
    """

    @staticmethod
    def _return_width(value, known_widths, module_name):
        """Infer tuple width, including ``(*other_handler(), *reset())``."""
        import ast

        if isinstance(value, ast.Tuple):
            width = 0
            for element in value.elts:
                if not isinstance(element, ast.Starred):
                    width += 1
                    continue
                call = element.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                    return None
                nested = known_widths.get((module_name, call.func.id))
                if nested is None:
                    return None
                width += nested
            return width
        if value is not None and not isinstance(value, ast.Constant):
            return 1
        return None

    @staticmethod
    def return_widths():
        """{(module, function): number of values returned}, from the source.

        Keyed by module because the two tabs define handlers of the same name
        with different shapes: on_open_working_directory returns five values in
        one tab and six in the other.
        """
        import ast
        import pathlib

        widths = {}
        for name in ("protein_md_simulation.py", "protein_ligand_complex_md_simulation.py"):
            tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("on_"):
                    continue
                found = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return):
                        value = sub.value
                    elif isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Yield):
                        value = sub.value.value
                    else:
                        continue
                    width = WiringTests._return_width(value, widths, name[:-3])
                    if width is not None:
                        found.add(width)
                # Only handlers whose exits already agree; the next test covers
                # the ones that do not.
                if len(found) == 1:
                    widths[(name[:-3], node.name)] = found.pop()
        return widths

    def test_every_exit_of_a_handler_returns_the_same_number_of_values(self):
        """An error path one value short breaks only when something goes wrong."""
        import ast
        import pathlib

        known_widths = self.return_widths()
        for name in ("protein_md_simulation.py", "protein_ligand_complex_md_simulation.py"):
            tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("on_"):
                    continue
                found = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return):
                        value = sub.value
                    elif isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Yield):
                        value = sub.value.value
                    else:
                        continue
                    width = self._return_width(value, known_widths, name[:-3])
                    if width is not None:
                        found.add(width)
                with self.subTest(module=name, handler=node.name):
                    self.assertLessEqual(len(found), 1,
                                         f"{node.name} exits with {sorted(found)} values")

    def test_every_handler_matches_the_outputs_it_is_wired_to(self):
        import webui

        widths = self.return_widths()
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        checked = 0
        for handler in handlers:
            key = (getattr(handler.fn, "__module__", ""), getattr(handler.fn, "__name__", ""))
            if key not in widths or not handler.outputs:
                continue
            with self.subTest(module=key[0], handler=key[1]):
                self.assertEqual(len(handler.outputs), widths[key])
            checked += 1

        self.assertGreater(checked, 50, "expected to check most of the wired handlers")

    def test_initial_and_continuation_controls_share_one_production_state(self):
        """The two buttons target one -deffnm prefix and therefore one live job."""
        import webui

        handlers = list(webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                        else webui.blocks.fns)
        for module_name in ("protein_md_simulation",
                            "protein_ligand_complex_md_simulation"):
            by_name = {
                handler.fn.__name__: handler
                for handler in handlers
                if getattr(handler.fn, "__module__", "") == module_name
                and getattr(handler.fn, "__name__", "") in {
                    "on_run_prod_md", "on_continue_prod_md"}
            }
            with self.subTest(module=module_name):
                self.assertEqual(set(by_name), {"on_run_prod_md", "on_continue_prod_md"})
                initial = by_name["on_run_prod_md"]
                continuation = by_name["on_continue_prod_md"]
                self.assertIs(initial.inputs[-1], continuation.inputs[-1])
                self.assertIs(initial.outputs[-2], continuation.outputs[-2])


class LigandUploadTests(WorkingDirectoryTestCase):
    """The analysis selects the ligand as "resname LIG", so the upload enforces it."""

    def upload(self, content, name="ligand.pdb", residue_name="LIG"):
        source = os.path.join(tempfile.mkdtemp(), name)
        self.addCleanup(shutil.rmtree, os.path.dirname(source), ignore_errors=True)
        with open(source, "w") as handle:
            handle.write(content)
        return complex_workflow.on_upload_ligand_structure_file(
            self.working_directory_path, name, residue_name, source)

    def test_an_unk_ligand_is_stored_as_lig(self):
        files, status = self.upload(UNK_LIGAND_PDB)

        self.assertIn("ligand.pdb", files)
        with open(self.path("ligand.pdb")) as handle:
            stored = handle.read()
        self.assertNotIn("UNK", stored)
        self.assertIn("LIG", stored)

        plain = self.plain_text(status)
        self.assertIn("uploaded successfully", plain)
        self.assertIn("UNK", plain)                    # the rename is reported, not silent

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)

    def test_a_ligand_already_named_lig_uploads_unchanged(self):
        content = UNK_LIGAND_PDB.replace("UNK", "LIG")
        files, status = self.upload(content)

        self.assertIn("ligand.pdb", files)
        with open(self.path("ligand.pdb")) as handle:
            self.assertEqual(handle.read(), content)
        self.assertNotIn("renamed", self.plain_text(status))

    def test_a_ligand_with_an_empty_residue_field_is_named_lig(self):
        """The regression that made data/3BAJ_5a unanalysable.

        Real ligand PDBs come with columns 18-20 blank. Those records were being
        skipped by the guard that leaves a bare "TER" alone, so the file reached
        acpype unnamed, came back as UNK, and "resname LIG" then matched nothing.
        """
        blank = UNK_LIGAND_PDB.replace("UNK", "   ")
        files, status = self.upload(blank)

        with open(self.path("ligand.pdb")) as handle:
            stored = handle.read()
        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertIn("(blank)", self.plain_text(status))
        self.assertNotIn("HETATM    1  C1     ", stored)

    def test_the_named_residue_is_the_one_renamed(self):
        """A file holding more than the ligand: only the named residue moves."""
        mixed = (UNK_LIGAND_PDB.replace("END\n", "")
                 + "HETATM    3  O   HOH A 902      20.000  20.000  20.000  1.00  0.00           O\n"
                 + "END\n")
        files, status = self.upload(mixed, residue_name="UNK")

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertEqual(universe.select_atoms("resname HOH").n_atoms, 1)
        self.assertIn("UNK", self.plain_text(status))

    def test_a_name_absent_from_the_file_falls_back_to_renaming_everything(self):
        """An uploaded ligand file is the ligand, so the default still works when
        the residue is called something the user did not type."""
        files, status = self.upload(UNK_LIGAND_PDB.replace("UNK", "5A "), residue_name="LIG")

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertIn("5A", self.plain_text(status))

    def test_a_protein_upload_keeps_its_own_residue_names(self):
        source = write_structure_pdb(os.path.join(tempfile.mkdtemp(), "protein.pdb"))
        self.addCleanup(shutil.rmtree, os.path.dirname(source), ignore_errors=True)

        complex_workflow.on_upload_protein_structure_file(
            self.working_directory_path, "protein.pdb", source)

        with open(self.path("protein.pdb")) as handle:
            self.assertIn("GLY", handle.read())


class LigandTopologyGenerationTests(WorkingDirectoryTestCase):
    def test_acpype_position_restraints_are_copied_with_the_itp(self):
        def fake_acpype(_cmd, cwd):
            ligand_directory = os.path.join(cwd, "drug.acpype")
            os.makedirs(ligand_directory)
            outputs = {
                "drug_GMX.gro": "Ligand\n0\n1.0 1.0 1.0\n",
                "drug_GMX.itp": "[ moleculetype ]\nDrug_X 3\n",
                "posre_drug.itp": "[ position_restraints ]\n1 1 1000 1000 1000\n",
            }
            for file_name, content in outputs.items():
                with open(os.path.join(ligand_directory, file_name), "w") as handle:
                    handle.write(content)

        with unittest.mock.patch.object(
            complex_workflow, "run_checked_command", side_effect=fake_acpype
        ):
            files, status = complex_workflow.on_generate_ligand_topology(
                self.working_directory_path, "drug.pdb", "drug", 0, "bcc", "gaff2"
            )

        self.assertIn("drug_GMX.gro", files)
        self.assertIn("drug_GMX.itp", files)
        self.assertIn("posre_drug.itp", files)
        self.assertTrue(os.path.isdir(self.path("drug.acpype")))
        self.assertIn("successfully", self.plain_text(status))
        with open(self.path("posre_drug.itp")) as handle:
            self.assertIn("[ position_restraints ]", handle.read())


class IonAdditionContractTests(WorkingDirectoryTestCase):
    class SuccessfulGenion:
        returncode = 0

        def __init__(self, command, **_kwargs):
            self.command = command
            with open(command[command.index("-o") + 1], "w") as handle:
                handle.write("ions\n0\n1 1 1\n")

        def communicate(self, input):
            return "", ""

    def setUp(self):
        super().setUp()
        with open(self.path("input.top"), "w") as handle:
            handle.write('#include "amber99sb-ildn.ff/forcefield.itp"\n')

    def run_ions(self, module, mode="Concentration", concentration=150,
                 cation_charge=1, anion_charge=-1,
                 number_of_cations=5, number_of_anions=5,
                 neutralize=True, cation_name="NA", anion_name="CL"):
        captured = []

        def launch(command, **kwargs):
            captured.append(command)
            return self.SuccessfulGenion(command, **kwargs)

        with unittest.mock.patch.object(module, "_find_sol_group", return_value="13"), \
                unittest.mock.patch.object(module.subprocess, "Popen", side_effect=launch), \
                unittest.mock.patch.object(module, "run_checked_command") as validate:
            result = module.on_add_ions(
                self.working_directory_path, "ions.tpr", "ions.gro",
                "input.top", "ions.top", cation_name, anion_name, mode,
                concentration, cation_charge, anion_charge, number_of_cations,
                number_of_anions, neutralize)
        return captured, result

    def test_concentration_mode_passes_multivalent_ion_charges(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                commands, (_, status) = self.run_ions(
                    module, cation_name="MG", cation_charge=2,
                    anion_name="CL", anion_charge=-1)
                self.assertIn("successfully", self.plain_text(status))
                self.assertEqual(len(commands), 1)
                command = commands[0]
                self.assertEqual(command[command.index("-pq") + 1], "2")
                self.assertEqual(command[command.index("-nq") + 1], "-1")
                self.assertEqual(command[command.index("-conc") + 1], "0.15")
                self.assertNotIn("-np", command)
                self.assertNotIn("-nn", command)

    def test_number_mode_uses_exact_validated_counts_and_charges(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                commands, (_, status) = self.run_ions(
                    module, mode="Number", cation_name="CA", cation_charge=2,
                    anion_name="BR", anion_charge=-1,
                    number_of_cations=7, number_of_anions=11)
                self.assertIn("successfully", self.plain_text(status))
                command = commands[0]
                for option, value in (("-pq", "2"), ("-nq", "-1"),
                                      ("-np", "7"), ("-nn", "11")):
                    self.assertEqual(command[command.index(option) + 1], value)
                self.assertNotIn("-conc", command)

    def test_known_ion_names_reject_inconsistent_charges_before_subprocess(self):
        cases = []
        for name in ("NA", "K", "LI"):
            cases.append((name, 2, "CL", -1, name, "+1"))
        for name in ("MG", "CA", "ZN"):
            cases.append((name, 1, "CL", -1, name, "+2"))
        for name in ("CL", "F", "BR", "I"):
            cases.append(("NA", 1, name, -2, name, "-1"))

        for module in (workflow, complex_workflow):
            for mode in ("Concentration", "Number"):
                for (cation_name, cation_charge, anion_name, anion_charge,
                     expected_name, expected_charge) in cases:
                    with self.subTest(module=module.__name__, mode=mode,
                                      ion=expected_name):
                        commands, (_, status) = self.run_ions(
                            module, mode=mode,
                            cation_name=cation_name,
                            cation_charge=cation_charge,
                            anion_name=anion_name,
                            anion_charge=anion_charge)
                        self.assertEqual(commands, [])
                        message = self.plain_text(status)
                        self.assertIn(
                            f"ion residue name '{expected_name}'", message)
                        self.assertIn(
                            f"requires charge {expected_charge}", message)

    def test_unlisted_custom_ion_names_remain_available(self):
        for module in (workflow, complex_workflow):
            for mode in ("Concentration", "Number"):
                with self.subTest(module=module.__name__, mode=mode):
                    commands, (_, status) = self.run_ions(
                        module, mode=mode, cation_name="X3p",
                        cation_charge=3, anion_name="x2N", anion_charge=-2)
                    self.assertEqual(len(commands), 1)
                    command = commands[0]
                    self.assertEqual(command[command.index("-pname") + 1], "X3p")
                    self.assertEqual(command[command.index("-nname") + 1], "x2N")
                    self.assertEqual(command[command.index("-pq") + 1], "3")
                    self.assertEqual(command[command.index("-nq") + 1], "-2")
                    self.assertIn("successfully", self.plain_text(status))

    def test_ion_names_are_normalized_and_malformed_values_are_rejected(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__, case="normalization"):
                commands, (_, status) = self.run_ions(
                    module, cation_name="na", anion_name="cl")
                self.assertIn("successfully", self.plain_text(status))
                command = commands[0]
                self.assertEqual(command[command.index("-pname") + 1], "NA")
                self.assertEqual(command[command.index("-nname") + 1], "CL")

            for bad_name in ("", " NA", "SIXLET", "N A", "N\nA", None):
                with self.subTest(module=module.__name__, bad_name=bad_name):
                    commands, (_, status) = self.run_ions(
                        module, cation_name=bad_name)
                    self.assertEqual(commands, [])
                    self.assertIn("1 to 5", self.plain_text(status))

    def test_genion_success_is_not_published_when_topology_lacks_the_ion(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                with open(self.path("ions.gro"), "w") as handle:
                    handle.write("known-good structure")
                with open(self.path("ions.top"), "w") as handle:
                    handle.write("known-good topology")

                with unittest.mock.patch.object(
                        module, "_find_sol_group", return_value="13"), \
                        unittest.mock.patch.object(
                            module.subprocess, "Popen",
                            side_effect=self.SuccessfulGenion), \
                        unittest.mock.patch.object(
                            module, "run_checked_command",
                            side_effect=RuntimeError(
                                "No such moleculetype K")) as validate:
                    _, status = module.on_add_ions(
                        self.working_directory_path, "ions.tpr", "ions.gro",
                        "input.top", "ions.top", "K", "CL", "Number",
                        150, 1, -1, 2, 2, False)

                self.assertEqual(validate.call_count, 1)
                validation_command = validate.call_args.args[0]
                self.assertEqual(validation_command[1], "grompp")
                self.assertIn(".genion_stage_", validation_command[
                    validation_command.index("-c") + 1])
                self.assertIn("No such moleculetype K", self.plain_text(status))
                with open(self.path("ions.gro")) as handle:
                    self.assertEqual(handle.read(), "known-good structure")
                with open(self.path("ions.top")) as handle:
                    self.assertEqual(handle.read(), "known-good topology")

    def test_invalid_mode_and_numeric_values_never_reach_genion(self):
        cases = (
            ({"mode": "Approximate"}, "mode"),
            ({"concentration": float("nan")}, "concentration"),
            ({"concentration": 1001}, "concentration"),
            ({"cation_charge": 1.5}, "Cation charge"),
            ({"cation_charge": True}, "Cation charge"),
            ({"anion_charge": -1.5}, "Anion charge"),
            ({"mode": "Number", "number_of_cations": 2.5}, "Number of cations"),
            ({"mode": "Number", "number_of_anions": -1}, "Number of anions"),
            ({"neutralize": 1}, "Neutralize"),
        )
        for module in (workflow, complex_workflow):
            for arguments, expected in cases:
                with self.subTest(module=module.__name__, arguments=arguments), \
                        unittest.mock.patch.object(module, "_find_sol_group") as find, \
                        unittest.mock.patch.object(module.subprocess, "Popen") as launch:
                    _, status = module.on_add_ions(
                        self.working_directory_path, "ions.tpr", "ions.gro",
                        "input.top", "ions.top", "NA", "CL",
                        arguments.get("mode", "Concentration"),
                        arguments.get("concentration", 150),
                        arguments.get("cation_charge", 1),
                        arguments.get("anion_charge", -1),
                        arguments.get("number_of_cations", 5),
                        arguments.get("number_of_anions", 5),
                        arguments.get("neutralize", True))
                find.assert_not_called()
                launch.assert_not_called()
                self.assertIn(expected, self.plain_text(status))

    def test_hidden_mode_irrelevant_numeric_values_are_ignored(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__, mode="Concentration"):
                commands, (_, status) = self.run_ions(
                    module, mode="Concentration",
                    number_of_cations=2.5, number_of_anions=-100)
                self.assertEqual(len(commands), 1)
                self.assertIn("successfully", self.plain_text(status))

            with self.subTest(module=module.__name__, mode="Number"):
                commands, (_, status) = self.run_ions(
                    module, mode="Number", concentration=float("nan"))
                self.assertEqual(len(commands), 1)
                self.assertIn("successfully", self.plain_text(status))


class EnergyMinimizationHardwareTests(WorkingDirectoryTestCase):
    """GROMACS has no GPU PME for the minimisers, so this step must stay on the CPU."""

    def run_minimization(self, module, use_gpu):
        with unittest.mock.patch.object(module, "run_checked_command") as run:
            module.on_run_energy_minimization(self.working_directory_path, "em.tpr", 1, 4, use_gpu)
        return run.call_args.args[0]

    def test_the_command_never_asks_for_a_gpu(self):
        for module in (workflow, complex_workflow):
            for use_gpu in (True, False):
                with self.subTest(module=module.__name__, use_gpu=use_gpu):
                    cmd = self.run_minimization(module, use_gpu)
                    self.assertNotIn("gpu", cmd)
                    for task in ("-nb", "-pme", "-bonded"):
                        self.assertEqual(cmd[cmd.index(task) + 1], "cpu")


class ResourceValidationTests(WorkingDirectoryTestCase):
    """Slider bounds are presentation only; callbacks must distrust API input."""

    MDRUN_HANDLERS = (
        "on_run_energy_minimization",
        "on_run_nvt_equilibration",
        "on_run_npt_equilibration",
        "on_run_prod_md",
        "on_continue_prod_md",
    )

    def invoke(self, module, handler_name, mpi_rank, omp_threads,
               nnpot=False):
        handler = getattr(module, handler_name)
        if handler_name == "on_run_energy_minimization":
            return handler(
                self.working_directory_path, "md.tpr", mpi_rank, omp_threads,
                False)

        state = utils.ProcessStateDict()
        if handler_name in ("on_run_nvt_equilibration",
                            "on_run_npt_equilibration"):
            return handler(
                self.working_directory_path, "md.tpr", mpi_rank, omp_threads,
                False, state)
        if handler_name == "on_run_prod_md":
            return handler(
                self.working_directory_path, "md.tpr", mpi_rank, omp_threads,
                nnpot, False, state)
        return handler(
            self.working_directory_path, "md.tpr", "md.cpt", mpi_rank,
            omp_threads, nnpot, False, state)

    def test_every_mdrun_callback_rejects_nonpositive_resources_before_launch(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.MDRUN_HANDLERS:
                for mpi_rank, omp_threads, label in (
                        (0, 1, "MPI ranks"), (1, 0, "OpenMP threads")):
                    with self.subTest(module=module.__name__, handler=handler_name,
                                      label=label), \
                            unittest.mock.patch.object(
                                module, "run_checked_command") as checked, \
                            unittest.mock.patch.object(module, "subprocess") as process:
                        result = self.invoke(
                            module, handler_name, mpi_rank, omp_threads)
                    checked.assert_not_called()
                    process.Popen.assert_not_called()
                    status = result[1]
                    self.assertIn(label, self.plain_text(status))
                    self.assertIn("positive integer", self.plain_text(status))

    def test_resource_values_are_strict_integers_and_individually_bounded(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                for value in (True, 1.0, "1"):
                    with self.subTest(value=value), self.assertRaisesRegex(
                            ValueError, "positive integer"):
                        module._validate_positive_integer_resource(
                            value, "MPI ranks", 8)
                with self.assertRaisesRegex(ValueError, "server's limit of 8"):
                    module._validate_positive_integer_resource(
                        9, "MPI ranks", 8)
                with self.assertRaisesRegex(ValueError, "server's limit of 128"):
                    module._validate_positive_integer_resource(
                        129, "OpenMP threads", 128)

    def test_mpi_times_openmp_cannot_oversubscribe_logical_cpus(self):
        for module in (workflow, complex_workflow):
            def cpu_count(logical=True):
                return 4 if logical else 2

            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(
                        module.psutil, "cpu_count", side_effect=cpu_count):
                with self.assertRaisesRegex(
                        ValueError,
                        r"MPI ranks \(2\) × OpenMP threads \(3\).*6 CPU threads.*only 4"):
                    module._validate_mdrun_resources(2, 3)

    def test_nnpot_forces_one_rank_before_resource_validation(self):
        for module in (workflow, complex_workflow):
            for handler_name in ("on_run_prod_md", "on_continue_prod_md"):
                with self.subTest(module=module.__name__, handler=handler_name), \
                        unittest.mock.patch.object(
                            module, "_validate_mdrun_resources",
                            side_effect=ValueError("validation sentinel")) as validate, \
                        unittest.mock.patch.object(module, "subprocess") as process:
                    result = self.invoke(
                        module, handler_name, 10**9, 1, nnpot=True)
                validate.assert_called_once_with(1, 1)
                process.Popen.assert_not_called()
                self.assertIn("validation sentinel", self.plain_text(result[1]))


class GpuCheckboxTests(WorkingDirectoryTestCase):
    """Unticking "Use GPU" has to reach the CPU, not just stop asking for the GPU."""

    RUN_HANDLERS = ("on_run_nvt_equilibration", "on_run_npt_equilibration",
                    "on_run_prod_md", "on_continue_prod_md")

    def launch(self, module, handler_name, use_gpu, nnpot=False, mpi_rank=1):
        import inspect
        handler = getattr(module, handler_name)
        arguments = {}
        for name, parameter in inspect.signature(handler).parameters.items():
            if name == "working_directory_path":
                arguments[name] = self.working_directory_path
            elif name.endswith("process_state"):
                arguments[name] = utils.ProcessStateDict()
            elif name == "run_input_file_name":
                arguments[name] = "md.tpr"
            elif name == "checkpoint_file_name":
                arguments[name] = "md.cpt"
            elif name == "mpi_rank":
                arguments[name] = mpi_rank
            elif name == "use_gpu":
                arguments[name] = use_gpu
            elif name == "prod_md_nnpot_active":
                arguments[name] = nnpot
            elif parameter.annotation is bool:
                arguments[name] = False
            else:
                arguments[name] = 1

        if handler_name == "on_continue_prod_md":
            for name in ("md.tpr", "md.cpt"):
                with open(self.path(name), "wb") as handle:
                    handle.write(b"test")

        with unittest.mock.patch.object(module, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(module, "threading"):
            handler(**arguments)
            return fake_subprocess.Popen.call_args.args[0]

    def assertTaskAssignment(self, cmd, task, hardware):
        """Fail with the command rather than a ValueError when the option is absent."""
        self.assertIn(task, cmd, f"{task} left on auto, which picks a detected GPU: {cmd}")
        self.assertEqual(cmd[cmd.index(task) + 1], hardware, cmd)

    def test_unticking_the_box_pins_every_run_to_the_cpu(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.RUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd = self.launch(module, handler_name, use_gpu=False)
                    self.assertNotIn("gpu", cmd)
                    for task in ("-nb", "-pme", "-bonded"):
                        self.assertTaskAssignment(cmd, task, "cpu")

    def test_ticking_the_box_still_offloads(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.RUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd = self.launch(module, handler_name, use_gpu=True)
                    self.assertTaskAssignment(cmd, "-nb", "gpu")

    def test_multi_rank_production_does_not_request_gpu_update(self):
        """GROMACS cannot use the fully GPU-resident update with DD ranks."""
        for module in (workflow, complex_workflow):
            for handler_name in ("on_run_prod_md", "on_continue_prod_md"):
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd = self.launch(module, handler_name, use_gpu=True, mpi_rank=2)
                    self.assertTaskAssignment(cmd, "-nb", "gpu")
                    for option in ("-pme", "-bonded", "-update"):
                        self.assertNotIn(option, cmd)

    def test_a_neural_network_run_is_left_on_auto_when_the_box_is_ticked(self):
        """Its own offload set is unsafe here, but the model still wants the GPU."""
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.launch(module, "on_run_prod_md", use_gpu=True, nnpot=True)
                self.assertNotIn("-nb", cmd)
                self.assertNotIn("-pme", cmd)

    def test_a_neural_network_run_still_honours_an_unticked_box(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.launch(module, "on_run_prod_md", use_gpu=False, nnpot=True)
                self.assertTaskAssignment(cmd, "-nb", "cpu")


class MdrunWorkingDirectoryTests(WorkingDirectoryTestCase):
    """mdrun dumps step<n>b.pdb / step<n>c.pdb into its own working directory.

    Those names are hardcoded and no flag redirects them, so running from the
    repository root scatters crash dumps beside the source files.
    """

    MDRUN_HANDLERS = ("on_run_energy_minimization", "on_run_nvt_equilibration",
                      "on_run_npt_equilibration", "on_run_prod_md", "on_continue_prod_md")

    def invoke(self, module, handler_name):
        """Call one mdrun handler with placeholder arguments, capturing the launch."""
        import inspect
        handler = getattr(module, handler_name)
        arguments = {}
        for name, parameter in inspect.signature(handler).parameters.items():
            if name == "working_directory_path":
                arguments[name] = self.working_directory_path
            elif name.endswith("process_state"):
                arguments[name] = utils.ProcessStateDict()
            elif name == "run_input_file_name":
                arguments[name] = "md.tpr"
            elif name == "checkpoint_file_name":
                arguments[name] = "md.cpt"
            elif parameter.annotation is bool:
                arguments[name] = True
            else:
                arguments[name] = 1

        if handler_name == "on_continue_prod_md":
            for name in ("md.tpr", "md.cpt"):
                with open(self.path(name), "wb") as handle:
                    handle.write(b"test")

        with unittest.mock.patch.object(module, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(module, "run_checked_command") as run, \
                unittest.mock.patch.object(module, "threading"):
            handler(**arguments)
            if run.called:
                return run.call_args.args[0], run.call_args.kwargs.get("cwd")
            call = fake_subprocess.Popen.call_args
            return call.args[0], call.kwargs.get("cwd")

    def test_every_mdrun_launches_inside_the_job_directory(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.MDRUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd, cwd = self.invoke(module, handler_name)

                    self.assertEqual(cmd[:2], ["gmx", "mdrun"])
                    # path_security resolves the callback's directory argument.
                    self.assertEqual(cwd, os.path.abspath(self.working_directory_path))
                    # Plain names only: a path here would be resolved against the
                    # job directory a second time and miss.
                    self.assertEqual(cmd[cmd.index("-deffnm") + 1], "md")
                    if handler_name == "on_continue_prod_md":
                        self.assertEqual(cmd[cmd.index("-cpi") + 1], "md.cpt")
                        self.assertIn("-append", cmd)
                    elif handler_name == "on_run_prod_md":
                        # -deffnm also rewrites mdrun's default checkpoint-input
                        # name.  A fresh launch must explicitly point -cpi at a
                        # nonexistent guard path rather than discover md.cpt.
                        checkpoint_input = cmd[cmd.index("-cpi") + 1]
                        self.assertNotEqual(checkpoint_input, "md.cpt")
                        self.assertFalse(os.path.exists(checkpoint_input))
                        self.assertNotIn("-append", cmd)
                    else:
                        self.assertNotIn("-cpi", cmd)

    def test_fresh_production_does_not_discover_the_deffnm_checkpoint(self):
        # This is the exact collision that matters: mdrun treats md.cpt as the
        # implicit -cpi default after `-deffnm md` unless the callback overrides
        # it, turning the UI's fresh-start action into a continuation.
        with open(self.path("md.cpt"), "wb") as handle:
            handle.write(b"existing checkpoint")

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd, _ = self.invoke(module, "on_run_prod_md")
                checkpoint_input = cmd[cmd.index("-cpi") + 1]
                self.assertNotEqual(os.path.abspath(checkpoint_input),
                                    os.path.abspath(self.path("md.cpt")))
                self.assertFalse(os.path.exists(checkpoint_input))


if __name__ == "__main__":
    unittest.main()
