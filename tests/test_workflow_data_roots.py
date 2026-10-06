"""Regression tests for the independent workflow job roots."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import path_security
import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow


class ProteinWorkflowDataRootTests(unittest.TestCase):
    def setUp(self) -> None:
        path_security.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self._temporary_root = tempfile.TemporaryDirectory(
            prefix="_workflow_root_test_", dir=path_security.DATA_ROOT)
        self.addCleanup(self._temporary_root.cleanup)
        self.container = Path(self._temporary_root.name)
        self.protein_root = self.container / "protein_md"
        self.complex_root = self.container / "protein_ligand_complex_md"

    def test_open_creates_job_under_protein_root(self):
        with mock.patch.object(
                protein_workflow, "PROTEIN_MD_DATA_ROOT", self.protein_root):
            result = protein_workflow.on_open_working_directory("new_job")

        _, working_directory, files, _, _ = result
        self.assertEqual(Path(working_directory), self.protein_root / "new_job")
        self.assertTrue(Path(working_directory).is_dir())
        self.assertEqual(files, [])

    def test_picker_only_lists_protein_jobs(self):
        (self.protein_root / "Zulu").mkdir(parents=True)
        (self.protein_root / "alpha").mkdir()
        (self.protein_root / "not_a_job.txt").write_text(
            "not a directory", encoding="utf-8")
        (self.complex_root / "complex_job").mkdir(parents=True)

        with mock.patch.object(
                protein_workflow, "PROTEIN_MD_DATA_ROOT", self.protein_root):
            jobs = protein_workflow.get_working_directories()

        self.assertEqual(jobs, ["alpha", "Zulu"])
        self.assertNotIn("complex_job", jobs)

    def test_callbacks_reject_complex_workflow_directory(self):
        complex_root = path_security.DATA_ROOT / "protein_ligand_complex_md"
        complex_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="_wrong_workflow_", dir=complex_root) as directory:
            with self.assertRaises(ValueError):
                protein_workflow.on_clean_working_directory(directory)


class ProteinLigandComplexWorkflowDataRootTests(unittest.TestCase):
    def setUp(self) -> None:
        path_security.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self._temporary_root = tempfile.TemporaryDirectory(
            prefix="_workflow_root_test_", dir=path_security.DATA_ROOT)
        self.addCleanup(self._temporary_root.cleanup)
        self.container = Path(self._temporary_root.name)
        self.protein_root = self.container / "protein_md"
        self.complex_root = self.container / "protein_ligand_complex_md"

    def test_open_creates_job_under_complex_root(self):
        with mock.patch.object(
                complex_workflow, "WORKFLOW_DATA_ROOT", self.complex_root):
            result = complex_workflow.on_open_working_directory("new_job")

        _, working_directory, files, _, _, _ = result
        self.assertEqual(Path(working_directory), self.complex_root / "new_job")
        self.assertTrue(Path(working_directory).is_dir())
        self.assertEqual(files, [])

    def test_picker_only_lists_complex_jobs(self):
        (self.complex_root / "Zulu").mkdir(parents=True)
        (self.complex_root / "alpha").mkdir()
        (self.complex_root / "not_a_job.txt").write_text(
            "not a directory", encoding="utf-8")
        (self.protein_root / "protein_job").mkdir(parents=True)

        with mock.patch.object(
                complex_workflow, "WORKFLOW_DATA_ROOT", self.complex_root):
            jobs = complex_workflow.get_working_directories()

        self.assertEqual(jobs, ["alpha", "Zulu"])
        self.assertNotIn("protein_job", jobs)

    def test_callbacks_reject_protein_workflow_directory(self):
        protein_root = path_security.DATA_ROOT / "protein_md"
        protein_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="_wrong_workflow_", dir=protein_root) as directory:
            with self.assertRaises(ValueError):
                complex_workflow.on_clean_working_directory(directory)


if __name__ == "__main__":
    unittest.main()
