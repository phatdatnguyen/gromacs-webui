"""Regression tests for the managed NNPot index-group workflow."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

import gradio as gr

import protein_ligand_complex_md_simulation as complex_workflow
import utils
from .testing_support import WorkingDirectoryTestCase, requires_gromacs


def _completed(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def _write_six_atom_gro(
        path: str | Path, *, coordinate_shift: float = 0.0,
        velocity_shift: float = 0.0, box_length: float = 2.0,
        swap_first_atoms: bool = False,
        rename_first_atom: bool = False) -> None:
    atoms = [
        (1, "GLY", "N"),
        (1, "GLY", "CA"),
        (1, "GLY", "C"),
        (1, "GLY", "O"),
        (2, "LIG", "C1"),
        (2, "LIG", "O1"),
    ]
    if swap_first_atoms:
        atoms[0], atoms[1] = atoms[1], atoms[0]
    if rename_first_atom:
        resid, resname, _ = atoms[0]
        atoms[0] = (resid, resname, "NX")
    lines = ["NNPot identity fixture", str(len(atoms))]
    for atom_index, (resid, resname, atom_name) in enumerate(atoms, start=1):
        coordinate = coordinate_shift + atom_index * 0.01
        velocity = velocity_shift + atom_index * 0.001
        lines.append(
            f"{resid:5d}{resname:<5}{atom_name:>5}{atom_index:5d}"
            f"{coordinate:8.3f}{coordinate + 0.1:8.3f}"
            f"{coordinate + 0.2:8.3f}{velocity:8.4f}"
            f"{velocity + 0.01:8.4f}{velocity + 0.02:8.4f}"
        )
    lines.append(f"{box_length:10.5f}{box_length:10.5f}{box_length:10.5f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class NNPotIndexUtilityTests(WorkingDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        _write_six_atom_gro(self.path("complex.gro"))

    def _write_managed_index(self, structure_name: str = "complex.gro",
                             content: str | None = None) -> str:
        index_path = str(Path(self.path("nnpot.ndx")).resolve())
        Path(index_path).write_text(
            content or (
                "[ System ]\n1 2 3 4 5 6\n"
                "[ Protein ]\n1 2 3 4\n"
                "[ LIG ]\n5 6\n"
                "[ nnpot ]\n5 6\n"
            ),
            encoding="utf-8",
        )
        fingerprint, atom_count = \
            utils.get_nnpot_structure_atom_identity_fingerprint(
                str(Path(self.path(structure_name)).resolve()))
        utils._prepend_nnpot_index_metadata(
            index_path, fingerprint, atom_count, "Ligand")
        return index_path

    def test_managed_index_name_is_fixed(self):
        self.assertEqual(
            utils.get_nnpot_index_file_name("md_initial.mdp"),
            "nnpot.ndx",
        )
        self.assertEqual(
            utils.get_nnpot_index_file_name("production.v2.mdp"),
            "nnpot.ndx",
        )

    def test_ligand_group_preserves_defaults_and_renames_dynamic_last_group(self):
        calls: list[tuple[list[str], str | None, str | None]] = []

        def fake_runner(command, cwd=None, stdin_input=None, **_kwargs):
            command = list(command)
            calls.append((command, cwd, stdin_input))
            output_path = Path(command[command.index("-o") + 1])
            if "-n" not in command:
                # Three pre-existing defaults plus the group just made by
                # ``r LIG``. The implementation must discover index 3 rather
                # than relying on a topology-dependent hard-coded number.
                output_path.write_text(
                    "[ System ]\n1 2 3 4 5 6\n"
                    "[ Protein ]\n1 2 3 4\n"
                    "[ LIG ]\n5 6\n"
                    "[ LIG ]\n5 6\n",
                    encoding="utf-8",
                )
            else:
                output_path.write_text(
                    "[ System ]\n1 2 3 4 5 6\n"
                    "[ Protein ]\n1 2 3 4\n"
                    "[ LIG ]\n5 6\n"
                    "[ nnpot ]\n5 6\n",
                    encoding="utf-8",
                )
            return _completed(command)

        staged_directory = Path(self.working_directory_path) / ".stage"
        staged_directory.mkdir()
        output_path = str(staged_directory / "nnpot.ndx")
        atom_count, residue_count = utils.generate_nnpot_index_file(
            str(Path(self.path("complex.gro")).resolve()),
            str(Path(output_path).resolve()), "Ligand",
            str(Path(self.working_directory_path).resolve()),
            runner=fake_runner)

        self.assertEqual((atom_count, residue_count), (2, 0))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][0:2], ["gmx", "make_ndx"])
        self.assertEqual(calls[0][2], "r LIG\nq\n")
        self.assertIn("-n", calls[1][0])
        self.assertEqual(calls[1][2], "name 3 nnpot\nq\n")

        content = Path(output_path).read_text(encoding="utf-8")
        self.assertIn("; gromacs-webui-nnpot-schema = 1", content)
        self.assertRegex(
            content,
            r"(?m)^; gromacs-webui-nnpot-atom-identity-sha256 = "
            r"[0-9a-f]{64}$",
        )
        self.assertIn("; gromacs-webui-nnpot-atom-count = 6", content)
        self.assertIn("; gromacs-webui-nnpot-region = Ligand", content)
        self.assertIn("[ System ]", content)
        self.assertIn("[ Protein ]", content)
        self.assertIn("[ LIG ]", content)
        self.assertEqual(content.count("[ nnpot ]"), 1)
        self.assertRegex(content, r"(?s)\[ nnpot \]\s+5 6(?:\s|$)")

    def test_index_output_cannot_escape_the_job_through_a_symlink(self):
        outside = Path(self.working_directory_path).resolve().parent / (
            Path(self.working_directory_path).name + "_outside")
        outside.mkdir()
        self.addCleanup(
            lambda: outside.exists() and outside.rmdir())
        link = Path(self.working_directory_path) / ".escaped_stage"
        link.symlink_to(outside, target_is_directory=True)
        runner = mock.Mock()

        with self.assertRaisesRegex(ValueError, "inside the working directory"):
            utils.generate_nnpot_index_file(
                str(Path(self.path("complex.gro")).resolve()),
                str((link / "nnpot.ndx").resolve()), "Ligand",
                str(Path(self.working_directory_path).resolve()),
                runner=runner)

        runner.assert_not_called()
        self.assertFalse((outside / "nnpot.ndx").exists())

    def test_binding_residue_choice_uses_residue_com_selection_and_union(self):
        calls: list[tuple[list[str], str | None]] = []

        def fake_runner(command, cwd=None, stdin_input=None, **_kwargs):
            command = list(command)
            calls.append((command, stdin_input))
            if command[1] == "select":
                index_output = Path(command[command.index("-oi") + 1])
                index_output.write_text(
                    "# selected residue indices\n0.000 2 15 18\n",
                    encoding="utf-8",
                )
            else:
                output_path = Path(command[command.index("-o") + 1])
                if "-n" not in command:
                    output_path.write_text(
                        "[ System ]\n1 2 3 4 5 6\n"
                        "[ Protein ]\n1 2 3 4\n"
                        "[ LIG ]\n5 6\n"
                        "[ Protein_LIG ]\n1 2 5 6\n",
                        encoding="utf-8",
                    )
                else:
                    output_path.write_text(
                        "[ System ]\n1 2 3 4 5 6\n"
                        "[ Protein ]\n1 2 3 4\n"
                        "[ LIG ]\n5 6\n"
                        "[ nnpot ]\n1 2 5 6\n",
                        encoding="utf-8",
                    )
            return _completed(command)

        result = utils.generate_nnpot_index_file(
            str(Path(self.path("complex.gro")).resolve()),
            str(Path(self.path("nnpot.ndx")).resolve()),
            "Ligand and binding residues",
            str(Path(self.working_directory_path).resolve()),
            runner=fake_runner)

        self.assertEqual(result, (4, 2))
        self.assertEqual([call[0][1] for call in calls],
                         ["select", "make_ndx", "make_ndx"])
        selection_command = calls[0][0]
        selection = selection_command[selection_command.index("-select") + 1]
        self.assertIn('group "Protein"', selection)
        self.assertIn("within 0.5", selection)
        self.assertIn("resname LIG", selection)
        self.assertEqual(
            calls[1][1], "ri 15 18 | r LIG\nq\n")
        self.assertEqual(calls[2][1], "name 3 nnpot\nq\n")

    def test_region_choice_is_an_exact_server_side_whitelist(self):
        output_path = self.path("nnpot.ndx")
        Path(output_path).write_text("old index\n", encoding="utf-8")
        for choice in ("ligand", "resname LIG", "Protein", "", None):
            with self.subTest(choice=choice), self.assertRaisesRegex(
                    ValueError, "NNPot.*region|Ligand"):
                utils.generate_nnpot_index_file(
                    str(Path(self.path("complex.gro")).resolve()),
                    str(Path(output_path).resolve()), choice,
                    str(Path(self.working_directory_path).resolve()),
                    runner=mock.Mock())
            self.assertEqual(
                Path(output_path).read_text(encoding="utf-8"), "old index\n")

    def test_identity_provenance_allows_coordinate_velocity_and_box_changes(self):
        index_path = self._write_managed_index()
        _write_six_atom_gro(
            self.path("continued.gro"), coordinate_shift=1.25,
            velocity_shift=0.75, box_length=4.5)

        metadata = utils.validate_nnpot_index_for_structure(
            index_path, str(Path(self.path("continued.gro")).resolve()),
            str(Path(self.working_directory_path).resolve()))

        self.assertEqual(metadata["region"], "Ligand")
        self.assertEqual(metadata["atom-count"], 6)

    def test_identity_provenance_rejects_reordered_or_renamed_atoms(self):
        index_path = self._write_managed_index()
        variants = (
            ("reordered.gro", {"swap_first_atoms": True}),
            ("renamed.gro", {"rename_first_atom": True}),
        )
        for file_name, options in variants:
            with self.subTest(file_name=file_name):
                _write_six_atom_gro(self.path(file_name), **options)
                with self.assertRaisesRegex(
                        RuntimeError,
                        "different atom ordering or identity.*Regenerate"):
                    utils.validate_nnpot_index_for_structure(
                        index_path, str(Path(self.path(file_name)).resolve()),
                        str(Path(self.working_directory_path).resolve()))

    def test_identity_provenance_rejects_missing_and_malformed_metadata(self):
        index_path = str(Path(self.path("nnpot.ndx")).resolve())
        groups = "[ System ]\n1 2 3 4 5 6\n[ nnpot ]\n5 6\n"
        Path(index_path).write_text(groups, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "no complete.*Regenerate"):
            utils.validate_nnpot_index_for_structure(
                index_path, str(Path(self.path("complex.gro")).resolve()),
                str(Path(self.working_directory_path).resolve()))

        Path(index_path).write_text(
            "; gromacs-webui-nnpot-schema 1\n" + groups,
            encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "malformed.*Regenerate"):
            utils.validate_nnpot_index_for_structure(
                index_path, str(Path(self.path("complex.gro")).resolve()),
                str(Path(self.working_directory_path).resolve()))

    def test_validation_rejects_duplicate_nnpot_atoms_after_user_edit(self):
        index_path = self._write_managed_index(content=(
            "[ System ]\n1 2 3 4 5 6\n"
            "[ nnpot ]\n5 5 6\n"
        ))

        with self.assertRaisesRegex(RuntimeError, "duplicate atom indices"):
            utils.validate_nnpot_index_for_structure(
                index_path, str(Path(self.path("complex.gro")).resolve()),
                str(Path(self.working_directory_path).resolve()))

    @requires_gromacs
    def test_gromacs_accepts_managed_ndx_comment_metadata(self):
        index_path = self._write_managed_index()
        parsed_path = str(Path(self.path("parsed.ndx")).resolve())

        utils.run_checked_command(
            ["gmx", "make_ndx", "-n", index_path, "-o", parsed_path],
            cwd=str(Path(self.working_directory_path).resolve()),
            stdin_input="q\n")

        parsed_groups = utils._read_ndx_groups(parsed_path)
        self.assertEqual(parsed_groups[-1], ("nnpot", [5, 6]))


class NNPotMdpAndIndexHandlerTests(WorkingDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        Path(self.path("complex.gro")).write_text(
            "placeholder structure\n", encoding="utf-8")

    @staticmethod
    def _write_generated_index(_structure_path, output_path, _choice,
                               _working_directory_path, **_kwargs):
        Path(output_path).write_text(
            "[ System ]\n1 2\n[ nnpot ]\n1 2\n", encoding="utf-8")
        return 2, 0

    def _generate(self, **patches):
        with mock.patch.object(
                complex_workflow, "download_nnpot_model",
                return_value="/model-cache/ani2x.pt"), mock.patch.object(
                complex_workflow, "snapshot_nnpot_model_for_job",
                return_value=".nnpot_models/ani2x-test.pt"), mock.patch.object(
                complex_workflow, "generate_nnpot_index_file",
                side_effect=patches.get(
                    "index_side_effect", self._write_generated_index)) as generate:
            result = complex_workflow.on_generate_prod_md_mdp_file(
                self.working_directory_path, 1, 0.001, 300, 1.0,
                "Initial", -1, "md.mdp", True, "ani2x",
                "Ligand", "AMBER99SB-ILDN", "complex.gro")
        return result, generate

    def test_active_generation_publishes_companion_and_uses_exact_group_name(self):
        (files, status), generate = self._generate()

        index_name = utils.get_nnpot_index_file_name("md.mdp")
        self.assertIn("md.mdp", files)
        self.assertIn(index_name, files)
        self.assertIn("successfully", self.plain_text(status))
        mdp = Path(self.path("md.mdp")).read_text(encoding="utf-8")
        self.assertIn(
            f"nnpot-input-group     = {utils.NNPOT_INPUT_GROUP_NAME}", mdp)
        self.assertNotIn("nnpot-input-group     = Ligand", mdp)
        self.assertEqual(
            utils.read_nnpot_index_digest_from_mdp(self.path("md.mdp")),
            utils._sha256_file(self.path(index_name)))

        args = generate.call_args.args
        self.assertEqual(Path(args[0]), Path(self.path("complex.gro")).resolve())
        self.assertEqual(args[2], "Ligand")
        self.assertEqual(Path(args[3]), Path(self.working_directory_path).resolve())
        self.assertNotEqual(Path(args[1]), Path(self.path(index_name)).resolve())
        self.assertEqual(
            Path(args[1]).parent, Path(self.working_directory_path).resolve())

    def test_index_failure_preserves_the_previous_mdp_index_pair(self):
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        Path(self.path("md.mdp")).write_text("old mdp\n", encoding="utf-8")
        Path(self.path(index_name)).write_text("old index\n", encoding="utf-8")

        def fail_after_staging(_structure_path, output_path, *_args, **_kwargs):
            Path(output_path).write_text("new staged index\n", encoding="utf-8")
            raise RuntimeError("synthetic make_ndx failure")

        (_, status), _ = self._generate(index_side_effect=fail_after_staging)

        self.assertIn("color:red", status)
        self.assertIn("synthetic make_ndx failure", self.plain_text(status))
        self.assertEqual(
            Path(self.path("md.mdp")).read_text(encoding="utf-8"), "old mdp\n")
        self.assertEqual(
            Path(self.path(index_name)).read_text(encoding="utf-8"),
            "old index\n")


class NNPotGromppIndexTests(WorkingDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        Path(self.path("topol.top")).write_text(
            '#include "amber99sb-ildn.ff/forcefield.itp"\n',
            encoding="utf-8")
        Path(self.path("npt.gro")).write_text("structure\n", encoding="utf-8")

    def _write_mdp(self, active: bool) -> None:
        content = (
            "integrator = md\ncutoff-scheme = Verlet\n"
            "rlist = 1.0\nrvdw = 1.0\nrcoulomb = 1.0\n"
            "coulombtype = PME\nDispCorr = EnerPres\n"
            f"nnpot-active = {'true' if active else 'false'}\n"
        )
        if active:
            content += (
                "nnpot-modelfile = /custom/model.pt\n"
                f"nnpot-input-group = {utils.NNPOT_INPUT_GROUP_NAME}\n"
            )
        Path(self.path("md.mdp")).write_text(content, encoding="utf-8")

    def _run_with_patches(self):
        with mock.patch.object(
                complex_workflow, "get_nnpot_model_exporter_torch_version",
                return_value=None), mock.patch.object(
                complex_workflow, "get_nnpot_mdrun_environment",
                return_value={}), mock.patch.object(
                complex_workflow, "validate_nnpot_index_for_structure",
                return_value={}) as validate_index, mock.patch.object(
                complex_workflow, "validate_nnpot_input_group_charge",
                return_value=0.0) as validate_charge, mock.patch.object(
                complex_workflow, "record_nnpot_tpr_charge_attestation"), \
                mock.patch.object(
                    complex_workflow, "run_grompp_with_gromos_warning_policy",
                    return_value=None) as grompp:
            result = complex_workflow.on_generate_prod_md_tpr_file(
                self.working_directory_path, "npt.gro", "topol.top",
                "md.mdp", "md.tpr", 0, "AMBER99SB-ILDN", False)
        return result, validate_index, validate_charge, grompp

    def test_active_final_and_charge_probe_receive_the_same_companion_index(self):
        self._write_mdp(True)
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        Path(self.path(index_name)).write_text(
            "[ nnpot ]\n1 2\n", encoding="utf-8")
        parameter_path = Path(self.path("md.mdp"))
        parameter_path.write_text(
            utils.bind_nnpot_index_to_mdp_content(
                parameter_path.read_text(encoding="utf-8"),
                self.path(index_name)),
            encoding="utf-8")

        (_, status), validate_index, validate_charge, grompp = \
            self._run_with_patches()

        self.assertIn("successfully", self.plain_text(status))
        final_command = grompp.call_args.args[0]
        final_index = final_command[final_command.index("-n") + 1]
        self.assertEqual(final_index, str(Path(self.path(index_name)).resolve()))
        charge_call = validate_charge.call_args
        probe_index = charge_call.kwargs.get("index_file_path")
        if probe_index is None and len(charge_call.args) >= 7:
            probe_index = charge_call.args[6]
        self.assertEqual(probe_index, final_index)
        validate_index.assert_called_once_with(
            final_index, str(Path(self.path("npt.gro")).resolve()),
            str(Path(self.working_directory_path).resolve()))

    def test_active_mdp_requires_its_named_companion_before_grompp(self):
        self._write_mdp(True)

        (_, status), validate_index, validate_charge, grompp = \
            self._run_with_patches()

        self.assertIn("color:red", status)
        self.assertIn(
            utils.get_nnpot_index_file_name("md.mdp"),
            self.plain_text(status))
        validate_charge.assert_not_called()
        validate_index.assert_not_called()
        grompp.assert_not_called()

    def test_active_mdp_rejects_nnpot_index_generated_for_another_mdp(self):
        self._write_mdp(True)
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        index_path = Path(self.path(index_name))
        index_path.write_text("[ nnpot ]\n1 2\n", encoding="utf-8")
        parameter_path = Path(self.path("md.mdp"))
        parameter_path.write_text(
            utils.bind_nnpot_index_to_mdp_content(
                parameter_path.read_text(encoding="utf-8"), str(index_path)),
            encoding="utf-8")
        index_path.write_text("[ nnpot ]\n2\n", encoding="utf-8")

        (_, status), _, validate_charge, grompp = self._run_with_patches()

        self.assertIn("color:red", status)
        self.assertIn("does not match", self.plain_text(status))
        validate_charge.assert_not_called()
        grompp.assert_not_called()

    def test_active_mdp_requires_one_managed_index_digest(self):
        self._write_mdp(True)
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        Path(self.path(index_name)).write_text(
            "[ nnpot ]\n1 2\n", encoding="utf-8")

        (_, status), _, validate_charge, grompp = self._run_with_patches()

        self.assertIn("color:red", status)
        self.assertIn("exactly one managed index digest", self.plain_text(status))
        validate_charge.assert_not_called()
        grompp.assert_not_called()

    def test_active_mdp_rejects_malformed_managed_index_digest(self):
        self._write_mdp(True)
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        Path(self.path(index_name)).write_text(
            "[ nnpot ]\n1 2\n", encoding="utf-8")
        with Path(self.path("md.mdp")).open("a", encoding="utf-8") as handle:
            handle.write(
                f"; {utils.NNPOT_INDEX_DIGEST_COMMENT_KEY} = invalid\n")

        (_, status), _, validate_charge, grompp = self._run_with_patches()

        self.assertIn("color:red", status)
        self.assertIn("malformed managed index metadata", self.plain_text(status))
        validate_charge.assert_not_called()
        grompp.assert_not_called()

    def test_classical_mdp_ignores_a_stale_companion(self):
        self._write_mdp(False)
        index_name = utils.get_nnpot_index_file_name("md.mdp")
        Path(self.path(index_name)).write_text(
            "this stale file is intentionally invalid\n", encoding="utf-8")

        (_, status), validate_index, validate_charge, grompp = \
            self._run_with_patches()

        self.assertIn("successfully", self.plain_text(status))
        self.assertNotIn("-n", grompp.call_args.args[0])
        validate_charge.assert_not_called()
        validate_index.assert_not_called()

    def test_charge_probe_grompp_uses_the_supplied_index_file(self):
        self._write_mdp(True)
        index_path = self.path(utils.get_nnpot_index_file_name("md.mdp"))
        Path(index_path).write_text("[ nnpot ]\n1 2\n", encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run(command, cwd=None, **_kwargs):
            command = list(command)
            commands.append(command)
            if command[1] == "grompp":
                Path(command[command.index("-o") + 1]).write_bytes(b"tpr")
            elif command[1] == "select":
                Path(command[command.index("-on") + 1]).write_text(
                    "[ nnpot ]\n1 2\n", encoding="utf-8")
            return _completed(command)

        with mock.patch.object(
                utils, "run_checked_command", side_effect=fake_run), \
                mock.patch.object(
                    utils, "_stream_tpr_charges",
                    return_value={0: 0.4, 1: -0.4}):
            charge = utils.validate_nnpot_input_group_charge(
                self.working_directory_path, self.path("md.mdp"),
                self.path("npt.gro"), self.path("topol.top"), None, 0,
                index_file_path=str(Path(index_path).resolve()))

        self.assertAlmostEqual(charge, 0.0)
        grompp_command = next(command for command in commands
                              if command[1] == "grompp")
        self.assertEqual(
            grompp_command[grompp_command.index("-n") + 1],
            str(Path(index_path).resolve()))
        select_command = next(command for command in commands
                              if command[1] == "select")
        self.assertEqual(
            select_command[select_command.index("-n") + 1],
            str(Path(index_path).resolve()))


class NNPotInputGroupUiTests(unittest.TestCase):
    def test_ui_uses_the_exact_managed_region_dropdown(self):
        import webui

        components = [
            block for block in webui.blocks.blocks.values()
            if getattr(block, "label", None) == "NNPot Input Group"
        ]
        self.assertEqual(len(components), 1)
        dropdown = components[0]
        self.assertIsInstance(dropdown, gr.Dropdown)
        values = [choice[1] if isinstance(choice, (tuple, list)) else choice
                  for choice in dropdown.choices]
        self.assertEqual(values, list(utils.NNPOT_INDEX_REGION_CHOICES))
        self.assertEqual(dropdown.value, "Ligand")
        self.assertFalse(any(isinstance(component, gr.Textbox)
                             for component in components))


if __name__ == "__main__":
    unittest.main()
