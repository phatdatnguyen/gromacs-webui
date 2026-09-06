"""Cross-session exclusion for file mutations and synchronous minimisation."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow
import utils
from path_security import DATA_ROOT


class FakeRunningProcess:
    """The process-registry surface needed by these exclusion tests."""

    def poll(self):
        return None


class WorkingDirectoryCase(unittest.TestCase):
    def setUp(self) -> None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.mkdtemp(prefix="_mutation_concurrency_", dir=DATA_ROOT)
        self.registry_entries: list[tuple[str, object | None]] = []

    def tearDown(self) -> None:
        for key, process in reversed(self.registry_entries):
            utils.release_process_job(key, process)  # type: ignore[arg-type]
        shutil.rmtree(self.directory, ignore_errors=True)

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def reserve_running_process(self, prefix: str = "md") -> tuple[str, FakeRunningProcess]:
        key = utils.get_process_job_key(self.directory, prefix)
        claimed, active = utils.reserve_process_job(key)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        process = FakeRunningProcess()
        utils.register_process_job(key, process)  # type: ignore[arg-type]
        self.registry_entries.append((key, process))
        return key, process

    def remember_reservation(self, key: str) -> None:
        self.registry_entries.append((key, None))


class MaintenanceReservationTests(WorkingDirectoryCase):
    def test_maintenance_allows_only_its_own_managed_helper(self):
        own_key = utils.get_process_job_key(self.directory, ".managed-helper")
        other_key = utils.get_process_job_key(self.directory, "md")
        observed: list[tuple[bool, object | None]] = []

        with utils.reserve_working_directory_maintenance(self.directory):
            self.assertTrue(utils.is_working_directory_busy(self.directory))
            claimed, active = utils.reserve_process_job(own_key)
            self.assertTrue(claimed)
            self.assertIsNone(active)
            utils.release_process_job(own_key)

            worker = __import__("threading").Thread(
                target=lambda: observed.append(
                    utils.reserve_process_job(other_key)))
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

        self.assertEqual(observed, [(False, None)])
        self.assertFalse(utils.is_working_directory_busy(self.directory))
        claimed, active = utils.reserve_process_job(other_key)
        self.assertTrue(claimed)
        self.assertIsNone(active)
        self.remember_reservation(other_key)

    def test_maintenance_is_reentrant_until_the_outer_lease_exits(self):
        with utils.reserve_working_directory_maintenance(self.directory):
            with utils.reserve_working_directory_maintenance(self.directory):
                self.assertTrue(utils.is_working_directory_busy(self.directory))
            self.assertTrue(utils.is_working_directory_busy(self.directory))

        self.assertFalse(utils.is_working_directory_busy(self.directory))

    def test_live_process_atomically_blocks_maintenance(self):
        self.reserve_running_process()

        with self.assertRaises(utils.WorkingDirectoryBusyError):
            with utils.reserve_working_directory_maintenance(self.directory):
                self.fail("a live writer must prevent directory maintenance")

    def test_maintenance_lease_is_released_after_an_exception(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with utils.reserve_working_directory_maintenance(self.directory):
                raise RuntimeError("deliberate")

        with utils.reserve_working_directory_maintenance(self.directory):
            pass

    def test_read_lease_allows_its_own_helper_but_blocks_other_threads(self):
        own_key = utils.get_process_job_key(self.directory, ".analysis-helper")
        other_key = utils.get_process_job_key(self.directory, "md")
        observed: list[tuple[bool, object | None]] = []

        with utils.reserve_working_directory_read(self.directory):
            claimed, active = utils.reserve_process_job(own_key)
            self.assertTrue(claimed)
            self.assertIsNone(active)
            utils.release_process_job(own_key)

            worker = __import__("threading").Thread(
                target=lambda: observed.append(utils.reserve_process_job(other_key)))
            worker.start()
            worker.join(timeout=2)

        self.assertEqual(observed, [(False, None)])

    def test_live_writer_prevents_trajectory_analysis_and_result_loading(self):
        self.reserve_running_process()

        with mock.patch.object(protein_workflow.mda, "Universe") as universe:
            with self.assertRaisesRegex(utils.WorkingDirectoryBusyError, "still running"):
                protein_workflow.on_analyze_rmsd(
                    self.directory, "structure.pdb", "trajectory.xtc")
            universe.assert_not_called()

        with mock.patch.object(complex_workflow, "parse_mmpbsa_results") as parser:
            with self.assertRaisesRegex(utils.WorkingDirectoryBusyError, "still running"):
                complex_workflow.on_load_mmpbsa_results(
                    self.directory, complex_workflow.MMPBSA_RESULTS_FILE_NAME,
                    "structure.pdb", "trajectory.xtc", "mmpbsa.in")
            parser.assert_not_called()

    def test_streaming_analysis_lease_moves_with_generator_worker_thread(self):
        iterator = protein_workflow.on_analyze_sasa(
            self.directory, "md.tpr", "md.xtc", "protein", "",
            0.14, "sasa.xvg", "sasa_residue.xvg")
        first = next(iterator)
        self.assertIn("Running", str(first[-1]))

        results: list[object] = []
        with mock.patch.object(
                protein_workflow, "run_checked_command",
                side_effect=RuntimeError("deliberate stop")) as run:
            worker = __import__("threading").Thread(
                target=lambda: results.append(next(iterator)))
            worker.start()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(run.call_count, 1)
        self.assertIn("deliberate stop", str(results[0]))
        iterator.close()
        self.assertFalse(utils.is_working_directory_busy(self.directory))


class BusyFileMutationTests(WorkingDirectoryCase):
    def setUp(self) -> None:
        super().setUp()
        for name, content in (
                ("doomed.gro", "keep structure"),
                ("protein.gro", "protein structure"),
                ("ligand.gro", "ligand structure"),
                ("protein.top", "protein topology"),
                ("ligand.itp", "ligand topology"),
                ("boxed.gro", "keep old box"),
                ("complex.gro", "keep old complex"),
                ("complex.top", "keep old topology"),
                ("run.tpr", "run input"),
                ("input.xtc", "trajectory"),
                ("fixed.xtc", "keep old trajectory"),
                ("notes.mdp", "keep parameters"),
                ("#topol.top.1#", "keep backup"),
                ("protein.pdb", "keep protein"),
                ("ligand.pdb", "keep ligand")):
            with open(self.path(name), "w", encoding="utf-8") as handle:
                handle.write(content)

        self.source_directory = tempfile.mkdtemp(prefix="gromacs_upload_source_")
        self.addCleanup(shutil.rmtree, self.source_directory, ignore_errors=True)
        self.protein_source = os.path.join(self.source_directory, "protein.pdb")
        self.ligand_source = os.path.join(self.source_directory, "ligand.pdb")
        for path in (self.protein_source, self.ligand_source):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("replacement")

        self.reserve_running_process()

    def assert_warning_says_running(self, warning: mock.Mock) -> None:
        self.assertTrue(warning.called)
        self.assertIn("running", str(warning.call_args.args[0]).lower())

    def test_delete_save_and_clean_preserve_files_in_both_workflows(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module.gr, "Warning") as warning:
                module.on_delete_file(self.directory, "doomed.gro")
                self.assert_warning_says_running(warning)
                warning.reset_mock()
                module.on_save_text_file(self.directory, "notes.mdp", "replacement")
                self.assert_warning_says_running(warning)
                warning.reset_mock()
                module.on_clean_working_directory(self.directory)
                self.assert_warning_says_running(warning)

                with open(self.path("doomed.gro"), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "keep structure")
                with open(self.path("notes.mdp"), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "keep parameters")
                self.assertTrue(os.path.isfile(self.path("#topol.top.1#")))

    def test_protein_upload_preserves_destination_in_both_workflows(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                _, status = module.on_upload_protein_structure_file(
                    self.directory, "protein.pdb", self.protein_source)
                self.assertIn("running", status.lower())
                with open(self.path("protein.pdb"), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "keep protein")

    def test_ligand_upload_preserves_destination(self):
        _, status = complex_workflow.on_upload_ligand_structure_file(
            self.directory, "ligand.pdb", "LIG", self.ligand_source)

        self.assertIn("running", status.lower())
        with open(self.path("ligand.pdb"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "keep ligand")

    def test_live_writer_blocks_box_and_merge_mutations(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__, operation="box"), \
                    mock.patch.object(module, "run_checked_command") as run:
                _, status = module.on_generate_simulation_box(
                    self.directory, "protein.gro", "boxed.gro", "cubic", 1.0,
                    "AMBER99SB-ILDN")
                run.assert_not_called()
                self.assertIn("running", status.lower())

        with mock.patch.object(
                complex_workflow, "merge_protein_ligand_structures") as merge:
            _, status = complex_workflow.on_merge_structures(
                self.directory, "protein.gro", "ligand.gro", "complex.gro")
            merge.assert_not_called()
            self.assertIn("running", status.lower())

        with mock.patch.object(
                complex_workflow, "merge_protein_ligand_topologies") as merge:
            _, status = complex_workflow.on_merge_topologies(
                self.directory, "protein.top", "ligand.itp", "complex.top")
            merge.assert_not_called()
            self.assertIn("running", status.lower())

        for name, expected in (
                ("boxed.gro", "keep old box"),
                ("complex.gro", "keep old complex"),
                ("complex.top", "keep old topology"),
                ("fixed.xtc", "keep old trajectory")):
            with open(self.path(name), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), expected)

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__, operation="trajectory"), \
                    mock.patch.object(module, "get_gmx_group_input") as groups, \
                    mock.patch.object(module, "run_checked_command") as run:
                _, status = module.on_make_molecule_whole(
                    self.directory, "run.tpr", "input.xtc", "fixed.xtc")
                groups.assert_not_called()
                run.assert_not_called()
                self.assertIn("running", status.lower())
        with open(self.path("fixed.xtc"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "keep old trajectory")

    def _mdp_writers(self, module):
        force_field = "AMBER99SB-ILDN"
        return (
            ("ions.mdp", lambda: module.on_generate_ions_mdp_file(
                self.directory, "ions.mdp", force_field)),
            ("em.mdp", lambda: module.on_generate_energy_minimization_mdp_file(
                self.directory, "em.mdp", force_field)),
            ("nvt.mdp", lambda: module.on_generate_nvt_equilibration_mdp_file(
                self.directory, 1, 0.002, 300, "nvt.mdp", force_field)),
            ("npt.mdp", lambda: module.on_generate_npt_equilibration_mdp_file(
                self.directory, 1, 0.002, 300, 1.0, "npt.mdp", force_field)),
            ("prod.mdp", lambda: module.on_generate_prod_md_mdp_file(
                self.directory, 1, 0.002, 300, 1.0, "Initial", -1,
                "prod.mdp", False, "ani2x", "Protein", force_field)),
        )

    def test_live_writer_blocks_every_direct_mdp_writer(self):
        for module in (protein_workflow, complex_workflow):
            for file_name, invoke in self._mdp_writers(module):
                with self.subTest(module=module.__name__, file=file_name):
                    original = f"keep {module.__name__} {file_name}"
                    with open(self.path(file_name), "w", encoding="utf-8") as handle:
                        handle.write(original)

                    _, status = invoke()

                    self.assertIn("running", status.lower())
                    with open(self.path(file_name), encoding="utf-8") as handle:
                        self.assertEqual(handle.read(), original)

    def test_live_writer_blocks_mmpbsa_input_and_dataframe_exports(self):
        destinations = ("mmpbsa.in", "analysis.csv")
        for file_name in destinations:
            with open(self.path(file_name), "w", encoding="utf-8") as handle:
                handle.write(f"keep {file_name}")

        _, status = complex_workflow.on_generate_mmpbsa_input_file(
            self.directory, "mmpbsa.in", "1", "0", 1, 0.15, 300.0,
            ["MM-GBSA"], True, 2, "within 6")
        self.assertIn("running", status.lower())

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                _, status = module.on_export_df(
                    self.directory, __import__("pandas").DataFrame({"x": [1]}),
                    "analysis.csv")
                self.assertIn("running", status.lower())

        for file_name in destinations:
            with open(self.path(file_name), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), f"keep {file_name}")


class AtomicDirectWriteTests(WorkingDirectoryCase):
    def test_text_replace_failure_preserves_destination_and_cleans_temp(self):
        destination = self.path("parameters.mdp")
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write("original")

        with mock.patch.object(utils.os, "replace", side_effect=OSError("failed")), \
                self.assertRaisesRegex(OSError, "failed"):
            utils.atomic_write_text_file(destination, "replacement")

        with open(destination, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "original")
        self.assertFalse(any(name.startswith(".atomic_text_")
                             for name in os.listdir(self.directory)))

    def test_csv_generation_failure_preserves_destination_and_cleans_temp(self):
        destination = self.path("analysis.csv")
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write("original")

        class FailingFrame:
            def to_csv(self, path, index=False):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("partial")
                raise OSError("failed")

        with self.assertRaisesRegex(OSError, "failed"):
            utils.atomic_write_dataframe_csv(destination, FailingFrame())

        with open(destination, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "original")
        self.assertFalse(any(name.startswith(".atomic_csv_")
                             for name in os.listdir(self.directory)))

    def test_box_command_failure_preserves_destination_and_cleans_stage(self):
        with open(self.path("input.gro"), "w", encoding="utf-8") as handle:
            handle.write("input")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                with open(self.path("boxed.gro"), "w", encoding="utf-8") as handle:
                    handle.write("known-good box")

                def fail_after_partial_output(command, **_kwargs):
                    output = command[command.index("-o") + 1]
                    with open(output, "w", encoding="utf-8") as handle:
                        handle.write("partial")
                    raise RuntimeError("editconf failed")

                with mock.patch.object(
                        module, "run_checked_command",
                        side_effect=fail_after_partial_output):
                    _, status = module.on_generate_simulation_box(
                        self.directory, "input.gro", "boxed.gro", "cubic", 1.0,
                        "AMBER99SB-ILDN")

                self.assertIn("editconf failed", status)
                with open(self.path("boxed.gro"), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "known-good box")
                self.assertFalse(any(name.startswith(".box_stage_")
                                     for name in os.listdir(self.directory)))

    def test_trajectory_failure_preserves_destination_and_cleans_stage(self):
        for name in ("run.tpr", "input.xtc"):
            with open(self.path(name), "wb") as handle:
                handle.write(b"input")

        handlers = (
            "on_make_molecule_whole", "on_center_protein", "on_fit_backbone")
        for module in (protein_workflow, complex_workflow):
            for handler_name in handlers:
                with self.subTest(module=module.__name__, handler=handler_name):
                    with open(self.path("fixed.xtc"), "wb") as handle:
                        handle.write(b"known-good trajectory")

                    def fail_after_partial_output(command, **_kwargs):
                        output = command[command.index("-o") + 1]
                        with open(output, "wb") as handle:
                            handle.write(b"partial")
                        raise RuntimeError("trjconv failed")

                    with mock.patch.object(
                            module, "get_gmx_group_input", return_value="0\n"), \
                            mock.patch.object(
                                module, "run_checked_command",
                                side_effect=fail_after_partial_output):
                        _, status = getattr(module, handler_name)(
                            self.directory, "run.tpr", "input.xtc", "fixed.xtc")

                    self.assertIn("trjconv failed", status)
                    with open(self.path("fixed.xtc"), "rb") as handle:
                        self.assertEqual(handle.read(), b"known-good trajectory")
                    self.assertFalse(any(name.startswith(".trajectory_stage_")
                                         for name in os.listdir(self.directory)))


class EnergyMinimisationReservationTests(WorkingDirectoryCase):
    def setUp(self) -> None:
        super().setUp()
        with open(self.path("em.tpr"), "w", encoding="utf-8") as handle:
            handle.write("placeholder")

    def test_existing_writer_prevents_duplicate_minimisation(self):
        self.reserve_running_process("em")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, "run_checked_command") as run:
                _, status = module.on_run_energy_minimization(
                    self.directory, "em.tpr", 1, 1, False)
                run.assert_not_called()
                self.assertIn("already using this output", status)

    def test_minimisation_holds_then_releases_its_output_reservation(self):
        key = utils.get_process_job_key(self.directory, "em")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                def observe_reservation(*args, **kwargs):
                    self.assertTrue(utils.is_working_directory_busy(self.directory))
                    claimed, active = utils.reserve_process_job(key)
                    self.assertFalse(claimed)
                    self.assertIsNone(active)

                with mock.patch.object(
                        module, "run_checked_command", side_effect=observe_reservation):
                    _, status = module.on_run_energy_minimization(
                        self.directory, "em.tpr", 1, 1, False)
                self.assertIn("completed successfully", status)

                claimed, active = utils.reserve_process_job(key)
                self.assertTrue(claimed)
                self.assertIsNone(active)
                utils.release_process_job(key)

    def test_minimisation_releases_reservation_after_command_failure(self):
        key = utils.get_process_job_key(self.directory, "em")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), mock.patch.object(
                    module, "run_checked_command", side_effect=RuntimeError("failed")):
                _, status = module.on_run_energy_minimization(
                    self.directory, "em.tpr", 1, 1, False)
                self.assertIn("failed", status)

                claimed, active = utils.reserve_process_job(key)
                self.assertTrue(claimed)
                self.assertIsNone(active)
                utils.release_process_job(key)

    def test_directory_maintenance_prevents_minimisation_start(self):
        for module in (protein_workflow, complex_workflow):
            results: list[tuple[list[str], str]] = []
            with self.subTest(module=module.__name__), \
                    utils.reserve_working_directory_maintenance(self.directory), \
                    mock.patch.object(module, "run_checked_command") as run:
                worker = __import__("threading").Thread(
                    target=lambda: results.append(
                        module.on_run_energy_minimization(
                            self.directory, "em.tpr", 1, 1, False)))
                worker.start()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                run.assert_not_called()
                self.assertIn("already using this output", results[0][1])


if __name__ == "__main__":
    unittest.main()
