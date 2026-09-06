"""Containment checks for historical MM-PBSA result migration."""

from __future__ import annotations

import os
import tempfile
import unittest

import protein_ligand_complex_md_simulation as workflow
from .testing_support import WorkingDirectoryTestCase


class LegacyResultSecurityTests(WorkingDirectoryTestCase):
    def test_legacy_result_directory_cannot_be_a_symlink_outside_the_job(self):
        with tempfile.TemporaryDirectory() as outside:
            os.symlink(outside, self.path(workflow.MMPBSA_SUBDIRECTORY), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must stay inside the job"):
                workflow._legacy_mmpbsa_directory(self.working_directory_path)


if __name__ == "__main__":
    unittest.main()
