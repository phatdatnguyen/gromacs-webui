"""Tests for the per-analysis trajectory callbacks.

These are the first tests of the analysis handlers: the single on_analyze_md_traj
they replaced had no coverage, no error handling and no status output.
"""

from __future__ import annotations

import os
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
from .testing_support import WorkingDirectoryTestCase, write_structure_pdb, write_trajectory


class TrajectoryAnalysisTests(WorkingDirectoryTestCase):
    """Every analysis button: shape, status, and independence from pyplot."""

    FRAMES = 6

    def setUp(self):
        super().setUp()
        # A one-atom LIG residue is enough for a centre of mass and for the ligand
        # RMSD selection, and keeps the fixture buildable without GROMACS.
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=4, ions={"LIG": 1})
        write_trajectory(structure, self.path("traj.xtc"), n_frames=self.FRAMES)

    def analyse(self, handler):
        return handler(self.working_directory_path, "system.pdb", "traj.xtc")

    def assertAnalysisSucceeded(self, result, expected_columns):
        frame, figure, status = result
        text = self.plain_text(status)
        self.assertIn("successfully", text, text)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(list(frame.columns), expected_columns)
        self.assertFalse(frame.empty)
        self.assertIsInstance(figure, Figure)
        return frame

    def test_rmsd_runs_in_both_tabs(self):
        expected = {
            workflow: ["Time (ns)", "Protein RMSD (Å)"],
            complex_workflow: ["Time (ns)", "Protein RMSD (Å)", "Ligand RMSD (Å)"],
        }
        for module, columns in expected.items():
            with self.subTest(module=module.__name__):
                frame = self.assertAnalysisSucceeded(
                    self.analyse(module.on_analyze_rmsd), columns)
                self.assertEqual(len(frame), self.FRAMES)

    def test_rmsf_runs_in_both_tabs(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                frame = self.assertAnalysisSucceeded(
                    self.analyse(module.on_analyze_rmsf), ["Residue Index", "Cα RMSF (Å)"])
                # One row per C-alpha, i.e. per protein residue in the fixture.
                self.assertEqual(len(frame), 4)

    def test_minimum_distance_measures_the_closest_approach(self):
        frame = self.assertAnalysisSucceeded(
            self.analyse(complex_workflow.on_analyze_min_distance),
            ["Time (ns)", "Minimum distance (Å)"])

        self.assertEqual(len(frame), self.FRAMES)
        self.assertTrue((frame["Minimum distance (Å)"] > 0).all())

    def test_the_minimum_distance_never_exceeds_the_centre_of_mass_distance(self):
        """The closest pair of atoms cannot be further apart than the centres are
        when the ligand is a single atom at its own centre of mass."""
        minimum, _, _ = self.analyse(complex_workflow.on_analyze_min_distance)
        centres, _, _ = self.analyse(complex_workflow.on_analyze_com_distance)

        for closest, between_centres in zip(minimum["Minimum distance (Å)"],
                                            centres["Center of mass distance (Å)"]):
            self.assertLessEqual(closest, between_centres + 1e-6)

    def test_com_distance_builds_its_own_time_axis(self):
        """It used to borrow the time column from the RMSD result it shared a
        function with; separated, it has to derive its own."""
        frame = self.assertAnalysisSucceeded(
            self.analyse(complex_workflow.on_analyze_com_distance),
            ["Time (ns)", "Center of mass distance (Å)"])

        self.assertEqual(len(frame), self.FRAMES)
        self.assertEqual(len(frame["Time (ns)"]), len(frame["Center of mass distance (Å)"]))

    def test_no_analysis_leaves_a_figure_behind_in_pyplot(self):
        """pyplot's current figure is process-wide, so a handler that used it would
        let two concurrent analyses draw into each other."""
        before = plt.get_fignums()
        for module in (workflow, complex_workflow):
            self.analyse(module.on_analyze_rmsd)
            self.analyse(module.on_analyze_rmsf)
        self.analyse(complex_workflow.on_analyze_com_distance)
        self.analyse(complex_workflow.on_analyze_min_distance)

        self.assertEqual(plt.get_fignums(), before)

    def test_each_analysis_is_independent_of_the_others(self):
        """The point of the split: one button failing must not blank the rest."""
        good = self.analyse(workflow.on_analyze_rmsd)
        bad = workflow.on_analyze_rmsf(self.working_directory_path, "system.pdb", "absent.xtc")

        self.assertIn("successfully", self.plain_text(good[2]))
        self.assertIn("Error", self.plain_text(bad[2]))


class AnalysisErrorTests(WorkingDirectoryTestCase):
    """A bad input must come back as a red status, not an exception into Gradio."""

    HANDLERS = ("on_analyze_rmsd", "on_analyze_rmsf")

    def setUp(self):
        super().setUp()
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=4, ions={"LIG": 1})
        write_trajectory(structure, self.path("traj.xtc"), n_frames=4)

    def test_a_missing_trajectory_is_reported_not_raised(self):
        for module in (workflow, complex_workflow):
            for name in self.HANDLERS:
                with self.subTest(module=module.__name__, handler=name):
                    frame, figure, status = getattr(module, name)(
                        self.working_directory_path, "system.pdb", "absent.xtc")
                    self.assertIsNone(frame)
                    self.assertIsNone(figure)
                    self.assertIn("Error", self.plain_text(status))

    def test_a_structure_without_a_ligand_names_the_problem(self):
        """The complex analyses select the ligand as LIG; say so when it is absent."""
        write_structure_pdb(self.path("noligand.pdb"), n_residues=4)
        write_trajectory(self.path("noligand.pdb"), self.path("noligand.xtc"), n_frames=3)

        for name in ("on_analyze_rmsd", "on_analyze_com_distance", "on_analyze_min_distance"):
            with self.subTest(handler=name):
                _, _, status = getattr(complex_workflow, name)(
                    self.working_directory_path, "noligand.pdb", "noligand.xtc")
                text = self.plain_text(status)
                self.assertIn("Error", text)
                self.assertIn("LIG", text)

    def test_both_distance_analyses_name_a_missing_protein(self):
        """They are siblings and should fail the same way. Only one had the
        check, so the other returned an all-NaN series under a green status."""
        write_structure_pdb(self.path("ligand_only.pdb"), n_residues=0, ions={"LIG": 2})
        write_trajectory(self.path("ligand_only.pdb"), self.path("ligand_only.xtc"), n_frames=3)

        for name in ("on_analyze_min_distance", "on_analyze_com_distance"):
            with self.subTest(handler=name):
                frame, figure, status = getattr(complex_workflow, name)(
                    self.working_directory_path, "ligand_only.pdb", "ligand_only.xtc")

                self.assertIsNone(frame)
                text = self.plain_text(status)
                self.assertIn("Error", text)
                self.assertIn("protein", text.lower())

    def test_a_structure_without_a_protein_names_the_problem(self):
        write_structure_pdb(self.path("ions.pdb"), n_residues=0, ions={"NA": 3})
        write_trajectory(self.path("ions.pdb"), self.path("ions.xtc"), n_frames=3)

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                _, _, status = module.on_analyze_rmsf(
                    self.working_directory_path, "ions.pdb", "ions.xtc")
                text = self.plain_text(status)
                self.assertIn("Error", text)
                self.assertIn("C-alpha", text)


class ExportGuardTests(WorkingDirectoryTestCase):
    def test_exporting_before_running_says_so_instead_of_crashing(self):
        """Eight export buttons make "clicked before running" easy to do."""
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                files, status = module.on_export_df(self.working_directory_path, None, "out.csv")
                self.assertIn("Run the analysis", self.plain_text(status))
                self.assertNotIn("out.csv", files)
                self.assertFalse(os.path.exists(self.path("out.csv")))

    def test_a_real_frame_still_exports(self):
        frame = pd.DataFrame({"Time (ns)": [0.0, 0.1], "RMSD": [0.0, 1.5]})
        files, status = workflow.on_export_df(self.working_directory_path, frame, "rmsd.csv")
        self.assertIn("rmsd.csv", files)
        self.assertIn("File exported", self.plain_text(status))


if __name__ == "__main__":
    unittest.main()
