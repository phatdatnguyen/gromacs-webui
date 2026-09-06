"""Uploads must not destroy a previously valid file when copying fails."""

from __future__ import annotations

import os
import tempfile
import unittest.mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow
from .testing_support import WorkingDirectoryTestCase


class AtomicUploadTests(WorkingDirectoryTestCase):
    def setUp(self):
        super().setUp()
        descriptor, self.source = tempfile.mkstemp(suffix=".pdb")
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(self.source) and os.remove(self.source))
        with open(self.source, "w") as handle:
            handle.write("new content")

    def test_failed_protein_upload_preserves_existing_destination(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                destination = self.path("protein.pdb")
                with open(destination, "w") as handle:
                    handle.write("known-good content")

                with unittest.mock.patch.object(
                        module.shutil, "copy2", side_effect=OSError("copy failed")):
                    _, status = module.on_upload_protein_structure_file(
                        self.working_directory_path, "protein.pdb", self.source)

                with open(destination) as handle:
                    self.assertEqual(handle.read(), "known-good content")
                self.assertIn("copy failed", self.plain_text(status))
                self.assertFalse(any(name.startswith(".upload_") for name in os.listdir(
                    self.working_directory_path)))

    def test_failed_ligand_upload_preserves_existing_destination(self):
        destination = self.path("ligand.pdb")
        with open(destination, "w") as handle:
            handle.write("known-good ligand")

        with unittest.mock.patch.object(
                complex_workflow.shutil, "copy2", side_effect=OSError("copy failed")):
            _, status = complex_workflow.on_upload_ligand_structure_file(
                self.working_directory_path, "ligand.pdb", "LIG", self.source)

        with open(destination) as handle:
            self.assertEqual(handle.read(), "known-good ligand")
        self.assertIn("copy failed", self.plain_text(status))


class AtomicWorkflowOutputTests(WorkingDirectoryTestCase):
    @staticmethod
    def write(path: str, content: str) -> None:
        with open(path, "w") as handle:
            handle.write(content)

    def assert_preserved(self, file_name: str, content: str) -> None:
        with open(self.path(file_name)) as handle:
            self.assertEqual(handle.read(), content)

    def test_publish_never_replaces_or_deletes_an_existing_directory(self):
        for index, module in enumerate((protein_workflow, complex_workflow)):
            with self.subTest(module=module.__name__):
                staged = self.path(f".staged_{index}.top")
                destination = self.path(f"custom_{index}.ff")
                sentinel = os.path.join(destination, "forcefield.itp")
                os.mkdir(destination)
                self.write(staged, "new output")
                self.write(sentinel, "keep me")

                with self.assertRaisesRegex(ValueError, "non-regular"):
                    module._publish_staged_files([(staged, destination)])

                self.assertTrue(os.path.isfile(staged))
                self.assert_preserved(
                    f"custom_{index}.ff/forcefield.itp", "keep me")

    def test_publish_preflights_remove_directories_before_moving_outputs(self):
        for index, module in enumerate((protein_workflow, complex_workflow)):
            with self.subTest(module=module.__name__):
                staged = self.path(f".staged_remove_{index}.top")
                destination = self.path(f"published_{index}.top")
                protected = self.path(f"protected_{index}.ff")
                sentinel = os.path.join(protected, "forcefield.itp")
                os.mkdir(protected)
                self.write(staged, "new output")
                self.write(sentinel, "keep me too")

                with self.assertRaisesRegex(ValueError, "non-regular"):
                    module._publish_staged_files(
                        [(staged, destination)], remove_files=[protected])

                self.assertTrue(os.path.isfile(staged))
                self.assertFalse(os.path.exists(destination))
                self.assert_preserved(
                    f"protected_{index}.ff/forcefield.itp", "keep me too")

    def test_failed_solvation_preserves_existing_gro_and_top(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                self.write(self.path("box.gro"), "input structure")
                self.write(
                    self.path("input.top"),
                    '#include "amber99sb-ildn.ff/tip3p.itp"\nold input topology',
                )
                self.write(self.path("solvated.gro"), "known-good structure")
                self.write(self.path("solvated.top"), "known-good topology")

                def fail_after_writing(command, cwd):
                    self.write(command[command.index("-o") + 1], "partial structure")
                    self.write(command[command.index("-p") + 1], "partial topology")
                    raise RuntimeError("solvate failed")

                with unittest.mock.patch.object(
                        module, "run_checked_command", side_effect=fail_after_writing):
                    _, status = module.on_solvate_protein(
                        self.working_directory_path, "box.gro", "solvated.gro",
                        "input.top", "solvated.top", "spc216.gro", "TIP3P")

                self.assertIn("solvate failed", self.plain_text(status))
                self.assert_preserved("solvated.gro", "known-good structure")
                self.assert_preserved("solvated.top", "known-good topology")
                self.assertFalse(any(name.startswith(".solvate_stage_")
                                     for name in os.listdir(self.working_directory_path)))

    def test_successful_solvation_publishes_both_outputs_together(self):
        self.write(self.path("box.gro"), "input structure")
        self.write(
            self.path("input.top"),
            '#include "amber99sb-ildn.ff/tip3p.itp"\nold input topology',
        )

        def succeed(command, cwd):
            self.write(command[command.index("-o") + 1], "new structure")
            with open(command[command.index("-p") + 1], "a") as handle:
                handle.write("\nnew solvent count")

        with unittest.mock.patch.object(
                protein_workflow, "run_checked_command", side_effect=succeed):
            files, status = protein_workflow.on_solvate_protein(
                self.working_directory_path, "box.gro", "solvated.gro",
                "input.top", "solvated.top", "spc216.gro", "TIP3P")

        self.assertIn("successfully", self.plain_text(status))
        self.assertIn("solvated.gro", files)
        self.assertIn("solvated.top", files)
        self.assert_preserved("solvated.gro", "new structure")
        with open(self.path("solvated.top")) as handle:
            self.assertIn("new solvent count", handle.read())

    def test_failed_genion_preserves_existing_gro_and_top(self):
        class FailedGenion:
            returncode = 1

            def __init__(self, command, **kwargs):
                self.command = command

            def communicate(self, input):
                AtomicWorkflowOutputTests.write(
                    self.command[self.command.index("-o") + 1], "partial ions")
                AtomicWorkflowOutputTests.write(
                    self.command[self.command.index("-p") + 1], "partial topology")
                return "", "genion failed"

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                self.write(self.path("input.top"), "input topology")
                self.write(self.path("ions.gro"), "known-good ions")
                self.write(self.path("ions.top"), "known-good topology")
                with unittest.mock.patch.object(module, "_find_sol_group", return_value="13"), \
                        unittest.mock.patch.object(
                            module.subprocess, "Popen", side_effect=FailedGenion):
                    _, status = module.on_add_ions(
                        self.working_directory_path, "ions.tpr", "ions.gro",
                        "input.top", "ions.top", "NA", "CL", "Concentration",
                        150.0, 1, -1, 0, 0, True)

                self.assertIn("genion failed", self.plain_text(status))
                self.assert_preserved("ions.gro", "known-good ions")
                self.assert_preserved("ions.top", "known-good topology")
                self.assertFalse(any(name.startswith(".genion_stage_")
                                     for name in os.listdir(self.working_directory_path)))

    def test_acpype_missing_required_artifact_publishes_nothing(self):
        self.write(self.path("drug_GMX.gro"), "known-good gro")
        self.write(self.path("drug_GMX.itp"), "known-good itp")

        def incomplete_acpype(_command, cwd):
            output = os.path.join(cwd, "drug.acpype")
            os.mkdir(output)
            self.write(os.path.join(output, "drug_GMX.gro"), "partial gro")

        with unittest.mock.patch.object(
                complex_workflow, "run_checked_command",
                side_effect=incomplete_acpype):
            _, status = complex_workflow.on_generate_ligand_topology(
                self.working_directory_path, "drug.pdb", "drug", 0,
                "bcc", "gaff2", "AMBER99SB-ILDN")

        self.assertIn("not created", self.plain_text(status))
        self.assert_preserved("drug_GMX.gro", "known-good gro")
        self.assert_preserved("drug_GMX.itp", "known-good itp")
        self.assertFalse(any(name.startswith(".acpype_stage_")
                             for name in os.listdir(self.working_directory_path)))

    def test_failed_pdb2gmx_preserves_the_complete_previous_output_set(self):
        self.write(self.path("protein.pdb"), "input")
        expected = {
            "protein.gro": "known-good gro",
            "topology.top": "known-good top",
            "posre.itp": "known-good restraints",
            "topology_Protein_chain_A.itp": "known-good chain",
        }
        for name, content in expected.items():
            self.write(self.path(name), content)

        def fail_after_partial_output(command, cwd):
            self.write(os.path.join(cwd, command[command.index("-o") + 1]),
                       "partial gro")
            self.write(os.path.join(cwd, command[command.index("-p") + 1]),
                       "partial top")
            self.write(os.path.join(cwd, "posre.itp"), "partial restraints")
            raise RuntimeError("pdb2gmx failed")

        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    unittest.mock.patch.object(
                        module, "run_checked_command",
                        side_effect=fail_after_partial_output):
                _, status = module.on_generate_protein_topology(
                    self.working_directory_path, "protein.pdb", "protein.gro",
                    "topology.top", "AMBER99SB-ILDN", "TIP3P",
                    module.DEFAULT_TERMINUS_CHOICE, module.DEFAULT_TERMINUS_CHOICE)

            self.assertIn("pdb2gmx failed", self.plain_text(status))
            for name, content in expected.items():
                self.assert_preserved(name, content)
            self.assertFalse(any(name.startswith(".pdb2gmx_stage_")
                                 for name in os.listdir(self.working_directory_path)))

    def test_genion_group_probes_use_unique_private_directories(self):
        class Probe:
            returncode = 0

            def __init__(self, command, **kwargs):
                self.command = command

            def communicate(self, input):
                return "", "Group    13 (            SOL) has  2955 elements\n"

        self.write(self.path("input.top"), "topology")
        command = ["gmx", "genion", "-s", self.path("ions.tpr"),
                   "-o", self.path("ions.gro"), "-p", self.path("input.top")]
        captured = []

        def popen(command, **kwargs):
            captured.append(command)
            return Probe(command, **kwargs)

        with unittest.mock.patch.object(
                protein_workflow.subprocess, "Popen", side_effect=popen):
            self.assertEqual(protein_workflow._find_sol_group(
                command, self.working_directory_path), "13")
            self.assertEqual(protein_workflow._find_sol_group(
                command, self.working_directory_path), "13")

        first_output = captured[0][captured[0].index("-o") + 1]
        second_output = captured[1][captured[1].index("-o") + 1]
        self.assertNotEqual(os.path.dirname(first_output), os.path.dirname(second_output))
        self.assertFalse(any(name.startswith(".probe_genion_")
                             for name in os.listdir(self.working_directory_path)))


if __name__ == "__main__":
    import unittest
    unittest.main()
