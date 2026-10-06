"""Focused tests for immutable per-job NNPot model snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import protein_ligand_complex_md_simulation as complex_workflow
import utils
from path_security import DATA_ROOT


class NNPotSnapshotTests(unittest.TestCase):
    def setUp(self):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.job = tempfile.TemporaryDirectory(
            prefix="nnpot_snapshot_job_", dir=DATA_ROOT)
        self.cache = tempfile.TemporaryDirectory(prefix="nnpot_snapshot_cache_")
        self.model_path = Path(self.cache.name) / "ani2x.pt"
        self.model_bytes = b"immutable torchscript model bytes\x00\x01"
        self.model_path.write_bytes(self.model_bytes)

    def tearDown(self):
        self.cache.cleanup()
        self.job.cleanup()

    @staticmethod
    def provenance(version: str = "2.8.0"):
        return (
            mock.patch.object(
                utils, "_read_nnpot_archive_provenance",
                return_value=(
                    utils.get_expected_nnpot_model_config("ani2x"),
                    {"torch": version, "torchani": "2.9.0"},
                )),
            mock.patch.object(
                utils, "_gromacs_provenance_versions",
                return_value=("2026.4", "2.11.0")),
        )

    def snapshot(self, version: str = "2.8.0") -> str:
        package_patch, gromacs_patch = self.provenance(version)
        with package_patch, gromacs_patch:
            return utils.snapshot_nnpot_model_for_job(
                self.job.name, "ani2x", str(self.model_path))

    def test_snapshot_is_content_addressed_relative_and_has_provenance(self):
        digest = hashlib.sha256(self.model_bytes).hexdigest()
        relative_path = self.snapshot()

        self.assertEqual(
            relative_path, f".nnpot_models/ani2x-{digest}.pt")
        snapshot_path = Path(self.job.name) / relative_path
        self.assertEqual(snapshot_path.read_bytes(), self.model_bytes)
        self.assertEqual(snapshot_path.stat().st_mode & 0o777, 0o400)
        manifest_path = snapshot_path.with_suffix(".json")
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o400)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_sha256"], digest)
        self.assertEqual(
            metadata["model_config"],
            utils.get_expected_nnpot_model_config("ani2x"))
        self.assertEqual(metadata["python_torch_version"], "2.8.0")
        self.assertEqual(metadata["gromacs_version"], "2026.4")
        self.assertEqual(metadata["gromacs_torch_version"], "2.11.0")
        self.assertEqual(
            manifest_path.read_text(encoding="utf-8"),
            json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        replacement = Path(self.cache.name) / "replacement.pt"
        replacement.write_bytes(b"new global cache model")
        os.replace(replacement, self.model_path)
        self.assertEqual(snapshot_path.read_bytes(), self.model_bytes)

    def test_in_place_cache_mutation_cannot_change_snapshot_inode(self):
        relative_path = self.snapshot()
        snapshot_path = Path(self.job.name) / relative_path

        self.model_path.write_bytes(b"mutated cache contents")

        self.assertEqual(snapshot_path.read_bytes(), self.model_bytes)
        self.assertFalse(os.path.samefile(self.model_path, snapshot_path))

    def test_existing_same_hash_artifacts_are_not_replaced(self):
        relative_path = self.snapshot("2.8.0")
        snapshot_path = Path(self.job.name) / relative_path
        manifest_path = snapshot_path.with_suffix(".json")
        snapshot_inode = snapshot_path.stat().st_ino
        manifest_inode = manifest_path.stat().st_ino
        original_manifest = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(self.snapshot("9.9.9"), relative_path)

        self.assertEqual(snapshot_path.stat().st_ino, snapshot_inode)
        self.assertEqual(manifest_path.stat().st_ino, manifest_inode)
        self.assertEqual(
            manifest_path.read_text(encoding="utf-8"), original_manifest)

    def test_verifier_rejects_a_modified_snapshot(self):
        relative_path = self.snapshot()
        snapshot_path = Path(self.job.name) / relative_path
        tampered = snapshot_path.with_name("tampered.tmp")
        tampered.write_bytes(b"tampered")
        snapshot_path.chmod(0o600)
        os.replace(tampered, snapshot_path)

        with self.assertRaisesRegex(RuntimeError, "was modified"):
            utils.verify_nnpot_model_snapshot(self.job.name, relative_path)

    def test_verifier_rejects_a_manifest_with_the_wrong_hash(self):
        relative_path = self.snapshot()
        manifest_path = (Path(self.job.name) / relative_path).with_suffix(
            ".json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["model_sha256"] = "0" * 64
        manifest_path.chmod(0o600)
        manifest_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            utils.verify_nnpot_model_snapshot(self.job.name, relative_path)

    def test_verifier_requires_the_provenance_manifest(self):
        relative_path = self.snapshot()
        manifest_path = (Path(self.job.name) / relative_path).with_suffix(
            ".json")
        manifest_path.unlink()

        with self.assertRaisesRegex(RuntimeError, "manifest.*missing"):
            utils.verify_nnpot_model_snapshot(self.job.name, relative_path)

    def test_exporter_version_comes_from_snapshot_not_current_environment(self):
        relative_path = self.snapshot("2.8.0")
        absolute_path = str(Path(self.job.name) / relative_path)

        self.assertEqual(
            utils.get_nnpot_model_exporter_torch_version(
                self.job.name, relative_path),
            "2.8.0",
        )
        self.assertEqual(
            utils.get_nnpot_model_exporter_torch_version(
                self.job.name, absolute_path),
            "2.8.0",
        )

    def test_bundled_legacy_cache_path_is_rejected(self):
        with self.assertRaisesRegex(
                RuntimeError, "mutable legacy cache path"):
            utils.get_nnpot_model_exporter_torch_version(
                self.job.name, str(self.model_path))

    def test_custom_model_path_remains_an_expert_workflow(self):
        self.assertIsNone(
            utils.require_nnpot_model_snapshot_for_bundled_model(
                self.job.name, "/custom/research-model.pt"))

    def test_snapshot_uses_archive_exporter_not_active_environment(self):
        relative_path = self.snapshot("2.7.9+cu126")
        manifest_path = (Path(self.job.name) / relative_path).with_suffix(
            ".json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["python_torch_version"], "2.7.9+cu126")
        self.assertEqual(
            metadata["package_versions"]["torch"], "2.7.9+cu126")

    def test_verifier_rejects_inconsistent_version_provenance(self):
        relative_path = self.snapshot()
        manifest_path = (Path(self.job.name) / relative_path).with_suffix(
            ".json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["python_torch_version"] = "forged"
        manifest_path.chmod(0o600)
        manifest_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "inconsistent version"):
            utils.verify_nnpot_model_snapshot(self.job.name, relative_path)

    def test_verifier_rejects_missing_torch_version_provenance(self):
        relative_path = self.snapshot()
        manifest_path = (Path(self.job.name) / relative_path).with_suffix(
            ".json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["package_versions"].pop("torch")
        metadata["python_torch_version"] = None
        manifest_path.chmod(0o600)
        manifest_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "inconsistent version"):
            utils.verify_nnpot_model_snapshot(self.job.name, relative_path)

    def test_tpr_inspection_verifies_managed_snapshot_reference(self):
        relative_path = self.snapshot()
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = Protein\n",
            encoding="utf-8")
        utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0)
        dump = (
            "  nnpot:\n"
            "    active = true\n"
            f"    modelfile = {relative_path}\n"
            "    input-group = Protein\n"
        )
        completed = subprocess.CompletedProcess(
            ["gmx", "dump"], 0, stdout=dump, stderr="")
        with mock.patch.object(
                utils, "run_checked_command", return_value=completed), \
                mock.patch.object(utils, "_validate_tpr_nnpot_contract"), \
                mock.patch.object(
                    utils, "verify_nnpot_model_snapshot",
                    wraps=utils.verify_nnpot_model_snapshot) as verify:
            self.assertEqual(
                utils.inspect_tpr_nnpot_configuration(
                    self.job.name, "md.tpr"),
                (True, relative_path))
        verify.assert_called_once_with(self.job.name, relative_path)

    def test_tpr_charge_attestation_is_required_and_hash_bound(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"first tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = Protein\n",
            encoding="utf-8")

        manifest_path = Path(utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 2.0e-5))

        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o400)
        self.assertAlmostEqual(
            utils.verify_nnpot_tpr_charge_attestation(
                self.job.name, "md.tpr", "Protein"),
            2.0e-5,
        )
        tpr_path.write_bytes(b"different tpr")
        with self.assertRaisesRegex(RuntimeError, "no trusted record"):
            utils.verify_nnpot_tpr_charge_attestation(
                self.job.name, "md.tpr", "Protein")

    def test_tpr_charge_attestation_rejects_a_charged_record(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"tpr")
        digest = hashlib.sha256(b"tpr").hexdigest()
        directory = Path(self.job.name) / utils.NNPOT_TPR_ATTESTATION_DIRECTORY
        directory.mkdir(mode=0o700)
        manifest_path = directory / utils._nnpot_tpr_attestation_artifact_name(
            digest, ".json")
        manifest_path.write_text(json.dumps({
            "schema_version": utils.NNPOT_TPR_ATTESTATION_SCHEMA_VERSION,
            "tpr_sha256": digest,
            "input_group": "Protein",
            "original_group_charge_e": 1.0,
            "neutrality_tolerance_e": utils.NNPOT_CHARGE_TOLERANCE_E,
        }), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "does not prove.*neutral"):
            utils.verify_nnpot_tpr_charge_attestation(
                self.job.name, "md.tpr", "Protein")

    def test_managed_tpr_attestation_snapshots_the_exact_nnpot_index(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"managed nnpot tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = nnpot\n",
            encoding="utf-8")
        working_index = Path(self.job.name) / utils.NNPOT_INDEX_FILE_NAME
        original_index = "[ System ]\n1 2\n[ nnpot ]\n1 2\n"
        working_index.write_text(original_index, encoding="utf-8")

        manifest_path = Path(utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0,
            index_file_path=str(working_index)))
        snapshot_path = Path(utils.get_nnpot_tpr_index_snapshot_path(
            self.job.name, "md.tpr", utils.NNPOT_INPUT_GROUP_NAME))

        self.assertEqual(snapshot_path.read_text(encoding="utf-8"), original_index)
        self.assertEqual(snapshot_path.stat().st_mode & 0o777, 0o400)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["index_sha256"], hashlib.sha256(
                original_index.encode("utf-8")).hexdigest())
        self.assertEqual(metadata["index_snapshot"], snapshot_path.name)

        working_index.write_text(
            "[ System ]\n1 2\n[ nnpot ]\n2\n", encoding="utf-8")
        self.assertEqual(
            Path(utils.get_nnpot_tpr_index_snapshot_path(
                self.job.name, "md.tpr", utils.NNPOT_INPUT_GROUP_NAME)),
            snapshot_path)
        self.assertEqual(snapshot_path.read_text(encoding="utf-8"), original_index)

    def test_managed_tpr_attestation_rejects_a_modified_index_snapshot(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"managed nnpot tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = nnpot\n",
            encoding="utf-8")
        working_index = Path(self.job.name) / utils.NNPOT_INDEX_FILE_NAME
        working_index.write_text(
            "[ System ]\n1 2\n[ nnpot ]\n1 2\n", encoding="utf-8")
        utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0,
            index_file_path=str(working_index))
        snapshot_path = Path(utils.get_nnpot_tpr_index_snapshot_path(
            self.job.name, "md.tpr", utils.NNPOT_INPUT_GROUP_NAME))
        snapshot_path.chmod(0o600)
        snapshot_path.write_text("[ nnpot ]\n2\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "snapshot.*changed"):
            utils.get_nnpot_tpr_index_snapshot_path(
                self.job.name, "md.tpr", utils.NNPOT_INPUT_GROUP_NAME)

    def test_identical_tpr_reuses_attested_equivalent_index_snapshot(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"identical managed nnpot tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = nnpot\n",
            encoding="utf-8")
        working_index = Path(self.job.name) / utils.NNPOT_INDEX_FILE_NAME
        original_index = "[ System ]\n1 2\n[ nnpot ]\n1 2\n"
        working_index.write_text(original_index, encoding="utf-8")
        first_manifest = Path(utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0,
            index_file_path=str(working_index)))
        snapshot_path = Path(utils.get_nnpot_tpr_index_snapshot_path(
            self.job.name, "md.tpr", utils.NNPOT_INPUT_GROUP_NAME))
        original_snapshot_inode = snapshot_path.stat().st_ino

        working_index.write_text(
            original_index + "[ unused ]\n2\n", encoding="utf-8")
        second_manifest = Path(utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0,
            index_file_path=str(working_index)))

        self.assertEqual(second_manifest, first_manifest)
        self.assertEqual(snapshot_path.stat().st_ino, original_snapshot_inode)
        self.assertEqual(snapshot_path.read_text(encoding="utf-8"), original_index)

    def test_tpr_charge_attestation_rejects_a_different_input_group(self):
        tpr_path = Path(self.job.name) / "md.tpr"
        tpr_path.write_bytes(b"tpr")
        parameter_path = Path(self.job.name) / "md.mdp"
        parameter_path.write_text(
            "nnpot-active = true\nnnpot-input-group = Protein\n",
            encoding="utf-8")
        utils.record_nnpot_tpr_charge_attestation(
            self.job.name, "md.tpr", str(parameter_path), 0.0)

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            utils.verify_nnpot_tpr_charge_attestation(
                self.job.name, "md.tpr", "Protein_LIG")

    def test_complex_mdp_handler_references_job_snapshot(self):
        job = tempfile.TemporaryDirectory(
            prefix="nnpot_handler_", dir=DATA_ROOT)
        self.addCleanup(job.cleanup)
        (Path(job.name) / "complex.gro").write_text(
            "placeholder structure\n", encoding="utf-8")

        def write_index(_structure, output, _choice, _directory):
            Path(output).write_text(
                "[ System ]\n1 2\n[ nnpot ]\n1 2\n", encoding="utf-8")
            return 2, 0

        package_patch, gromacs_patch = self.provenance()
        with mock.patch.object(
                complex_workflow, "download_nnpot_model",
                return_value=str(self.model_path)), \
                mock.patch.object(
                    complex_workflow, "generate_nnpot_index_file",
                    side_effect=write_index), \
                package_patch, gromacs_patch:
            _, status = complex_workflow.on_generate_prod_md_mdp_file(
                job.name, 1, 0.001, 300, 1.0, "Initial", -1,
                "md.mdp", True, "ani2x", "Ligand",
                "AMBER99SB-ILDN", "complex.gro")

        self.assertIn("successfully", status)
        content = (Path(job.name) / "md.mdp").read_text(
            encoding="utf-8")
        digest = hashlib.sha256(self.model_bytes).hexdigest()
        self.assertIn(
            "nnpot-modelfile       = "
            f".nnpot_models/ani2x-{digest}.pt",
            content)
        self.assertNotIn(str(self.model_path), content)


if __name__ == "__main__":
    unittest.main()
