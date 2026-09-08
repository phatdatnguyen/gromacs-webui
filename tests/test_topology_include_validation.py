"""Regression tests for staged genion topology include resolution."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest

import utils
from .testing_support import WorkingDirectoryTestCase


class IonValidationIncludeBaseTests(WorkingDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.stage = tempfile.TemporaryDirectory(
            prefix=".genion_stage_", dir=self.working_directory_path)
        self.addCleanup(self.stage.cleanup)
        self.staged_topology = os.path.join(self.stage.name, "ions.top")
        self.staged_structure = os.path.join(self.stage.name, "ions.gro")

        os.makedirs(self.path("ligand/includes"))
        with open(self.path("ligand_GMX.itp"), "w") as handle:
            handle.write('#include "ligand/includes/parameters.itp"\n')
        with open(self.path("ligand/includes/parameters.itp"), "w") as handle:
            # A nested relative include deliberately walks back towards the job
            # root. GROMACS, rather than an incomplete app-side parser, remains
            # responsible for interpreting this graph.
            handle.write('#include "../../shared_atomtypes.itp"\n')
        with open(self.path("shared_atomtypes.itp"), "w") as handle:
            handle.write("[ atomtypes ]\n")

        self.topology_text = (
            '#include "amber99sb-ildn.ff/forcefield.itp"\n'
            '#include "ligand_GMX.itp"\n'
            "[ system ]\nIon validation regression\n"
        )
        with open(self.staged_topology, "w") as handle:
            handle.write(self.topology_text)
        with open(self.staged_structure, "w") as handle:
            handle.write("ions\n0\n1 1 1\n")

    def invoke(self, runner):
        return utils.validate_ionized_system_with_grompp(
            self.staged_structure,
            self.staged_topology,
            self.working_directory_path,
            runner=runner,
        )

    def test_grompp_topology_is_a_private_sibling_of_local_includes(self):
        captured_topology = None

        def runner(command, cwd):
            nonlocal captured_topology
            captured_topology = command[command.index("-p") + 1]
            self.assertEqual(
                os.path.dirname(captured_topology),
                os.path.realpath(self.working_directory_path),
            )
            self.assertTrue(
                os.path.basename(captured_topology).startswith(
                    "#gromacs_webui_validate_ions_"))
            self.assertNotEqual(
                os.path.realpath(captured_topology),
                os.path.realpath(self.staged_topology),
            )
            with open(captured_topology) as handle:
                self.assertEqual(handle.read(), self.topology_text)

            # These relative and nested paths resolve from the temporary
            # topology exactly as they did from the original job topology.
            ligand = os.path.join(
                os.path.dirname(captured_topology), "ligand_GMX.itp")
            nested = os.path.join(
                os.path.dirname(captured_topology),
                "ligand/includes/parameters.itp",
            )
            shared = os.path.normpath(os.path.join(
                os.path.dirname(nested), "../../shared_atomtypes.itp"))
            self.assertTrue(os.path.isfile(ligand))
            self.assertTrue(os.path.isfile(nested))
            self.assertTrue(os.path.isfile(shared))
            self.assertEqual(
                stat.S_IMODE(os.stat(captured_topology).st_mode) & 0o077,
                0,
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr="")

        self.assertIsNone(self.invoke(runner))
        self.assertIsNotNone(captured_topology)
        self.assertFalse(os.path.exists(captured_topology))
        self.assertFalse(any(
            name.startswith("#gromacs_webui_validate_ions_")
            for name in os.listdir(self.working_directory_path)
        ))

    def test_temporary_topology_is_removed_when_grompp_raises(self):
        captured_topology = None

        def runner(command, cwd):
            nonlocal captured_topology
            captured_topology = command[command.index("-p") + 1]
            self.assertTrue(os.path.isfile(captured_topology))
            raise RuntimeError("synthetic grompp failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic grompp failure"):
            self.invoke(runner)

        self.assertIsNotNone(captured_topology)
        self.assertFalse(os.path.exists(captured_topology))

    def test_staged_topology_outside_the_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            topology = os.path.join(outside, "ions.top")
            with open(topology, "w") as handle:
                handle.write(
                    '#include "amber99sb-ildn.ff/forcefield.itp"\n')

            with self.assertRaisesRegex(
                    ValueError, "inside its working directory"):
                utils.validate_ionized_system_with_grompp(
                    self.staged_structure,
                    topology,
                    self.working_directory_path,
                    runner=lambda *_args, **_kwargs: None,
                )


if __name__ == "__main__":
    unittest.main()
