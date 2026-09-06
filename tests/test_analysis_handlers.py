"""Tests for the per-analysis trajectory callbacks.

These are the first tests of the analysis handlers: the single on_analyze_md_traj
they replaced had no coverage, no error handling and no status output.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
import utils
from .testing_support import (WorkingDirectoryTestCase, pdb_line,
                              write_structure_pdb, write_trajectory)


def write_frames(structure_path: str, trajectory_path: str, frames: list[np.ndarray],
                 dimensions: np.ndarray | None = None) -> str:
    """Write explicit coordinates for analysis regressions."""
    universe = mda.Universe(structure_path)
    try:
        with mda.Writer(trajectory_path, universe.atoms.n_atoms) as writer:
            for positions in frames:
                universe.atoms.positions = positions
                universe.dimensions = dimensions
                writer.write(universe.atoms)
    finally:
        universe.trajectory.close()

    return trajectory_path


def rename_first_oxygen_as_sidechain(structure_path: str) -> None:
    """Give the small glycine fixture one atom outside the backbone selection."""
    with open(structure_path) as handle:
        lines = handle.readlines()

    for index, line in enumerate(lines):
        if line.startswith("ATOM") and line[12:16].strip() == "O":
            lines[index] = line[:12] + f"{'CB':<4}" + line[16:]
            break
    else:
        raise AssertionError("Fixture contains no oxygen to rename")

    with open(structure_path, "w") as handle:
        handle.writelines(lines)


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
                    self.analyse(module.on_analyze_rmsf),
                    ["Residue Index", "Residue", "Cα RMSF (Å)"])
                # One row per C-alpha, i.e. per protein residue in the fixture.
                self.assertEqual(len(frame), 4)

    def test_coordinate_only_rmsd_and_rmsf_report_the_pbc_limitation(self):
        for module in (workflow, complex_workflow):
            for handler in (module.on_analyze_rmsd, module.on_analyze_rmsf):
                with self.subTest(module=module.__name__, handler=handler.__name__):
                    _, _, status = self.analyse(handler)
                    self.assertIn("color:orange", status)
                    self.assertIn("no TPR", self.plain_text(status))

    def test_tpr_rmsd_uses_one_dynamic_backbone_fit_and_converts_nm_to_angstrom(self):
        with open(self.path("run.tpr"), "w") as handle:
            handle.write("mock TPR")

        cases = (
            (workflow, ["Backbone", "Protein"], [[0.1], [0.25]]),
            (complex_workflow, ["Backbone", "Protein", "LIG"],
             [[0.1, 0.4], [0.25, 0.7]]),
        )
        for module, expected_groups, rows in cases:
            with self.subTest(module=module.__name__):
                def write_rmsd_output(cmd, cwd=None, stdin_input=None):
                    output_name = cmd[cmd.index("-o") + 1]
                    with open(os.path.join(cwd, output_name), "w") as output:
                        output.write('@ xaxis label "Time (ns)"\n')
                        output.write('@ yaxis label "RMSD (nm)"\n')
                        for time, series in zip((0.0, 2.0), rows):
                            output.write(" ".join(map(str, [time, *series])) + "\n")

                with patch.object(module, "get_gmx_group_input",
                                  return_value="dynamic groups\n") as groups, \
                        patch.object(module, "run_checked_command",
                                     side_effect=write_rmsd_output) as run:
                    frame, _, status = module.on_analyze_rmsd(
                        self.working_directory_path, "system.pdb", "traj.xtc",
                        "run.tpr")

                self.assertIn("color:green", status)
                cmd = run.call_args.args[0]
                self.assertEqual(cmd[:2], ["gmx", "rms"])
                self.assertEqual(cmd[cmd.index("-s") + 1], "run.tpr")
                self.assertEqual(cmd[cmd.index("-f") + 1], "traj.xtc")
                self.assertEqual(cmd[cmd.index("-fit") + 1], "rot+trans")
                self.assertEqual(cmd[cmd.index("-pbc") + 1], "yes")
                self.assertEqual(cmd[cmd.index("-ng") + 1], str(len(rows[0])))
                self.assertEqual(groups.call_args.args[1], expected_groups)
                self.assertEqual(frame["Time (ns)"].tolist(), [0.0, 2.0])
                np.testing.assert_allclose(
                    frame.filter(like="RMSD").to_numpy(),
                    np.asarray(rows) * 10.0)
                self.assertFalse(any(
                    name.startswith(".rmsd_")
                    for name in os.listdir(self.working_directory_path)))

    def test_tpr_rmsd_rejects_a_decreasing_time_axis_and_cleans_up(self):
        with open(self.path("run.tpr"), "w") as handle:
            handle.write("mock TPR")

        def write_bad_output(cmd, cwd=None, stdin_input=None):
            output_name = cmd[cmd.index("-o") + 1]
            with open(os.path.join(cwd, output_name), "w") as output:
                output.write('@ xaxis label "Time (ns)"\n')
                output.write('@ yaxis label "RMSD (nm)"\n')
                output.write("2.0 0.1\n1.0 0.2\n")

        with patch.object(workflow, "get_gmx_group_input", return_value="4\n1\n"), \
                patch.object(workflow, "run_checked_command",
                             side_effect=write_bad_output):
            frame, figure, status = workflow.on_analyze_rmsd(
                self.working_directory_path, "system.pdb", "traj.xtc", "run.tpr")

        self.assertIsNone(frame)
        self.assertIsNone(figure)
        self.assertIn("decreasing time axis", self.plain_text(status))
        self.assertFalse(any(
            name.startswith(".rmsd_")
            for name in os.listdir(self.working_directory_path)))

    def test_tpr_rmsf_handlers_use_the_topology_aware_streaming_helper(self):
        with open(self.path("run.tpr"), "w") as handle:
            handle.write("mock TPR")
        expected = (np.array([1, 2]), np.array(["A:GLY1", "B:GLY1"]),
                    np.array([0.1, 0.2]))

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__), \
                    patch.object(module, "gromacs_topology_aware_ca_rmsf",
                                 return_value=expected) as calculate:
                frame, _, status = module.on_analyze_rmsf(
                    self.working_directory_path, "system.pdb", "traj.xtc", "run.tpr")

            calculate.assert_called_once_with(
                os.path.realpath(self.working_directory_path),
                "run.tpr", "traj.xtc", "system.pdb",
                group_resolver=module.get_gmx_group_input,
                command_runner=module.run_checked_command)
            self.assertIn("color:green", status)
            self.assertEqual(frame["Residue"].tolist(), ["A:GLY1", "B:GLY1"])

    def test_tpr_rmsf_clusters_protein_in_bounded_gromacs_chunks(self):
        with open(self.path("run.tpr"), "w") as handle:
            handle.write("mock TPR")

        def write_clustered_chunk(cmd, cwd=None, stdin_input=None):
            output_name = cmd[cmd.index("-o") + 1]
            source = mda.Universe(self.path("system.pdb"), self.path("traj.xtc"))
            try:
                protein = source.select_atoms("protein")
                with mda.Writer(os.path.join(cwd, output_name),
                                protein.n_atoms) as writer:
                    for _ in source.trajectory:
                        writer.write(protein)
            finally:
                source.trajectory.close()

        resolve = Mock(return_value="protein groups\n")
        run = Mock(side_effect=write_clustered_chunk)
        indices, labels, values = utils.gromacs_topology_aware_ca_rmsf(
            self.working_directory_path, "run.tpr", "traj.xtc", "system.pdb",
            group_resolver=resolve, command_runner=run)

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["gmx", "trjconv"])
        self.assertEqual(cmd[cmd.index("-pbc") + 1], "cluster")
        self.assertEqual(cmd[cmd.index("-ur") + 1], "compact")
        self.assertEqual(resolve.call_args.args[1], ["Protein", "Protein"])
        self.assertEqual(indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(len(labels), 4)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertFalse(any(
            "rmsf_cluster" in name
            for name in os.listdir(self.working_directory_path)))

    def test_topology_aware_rmsf_removes_multichain_image_jumps(self):
        atoms = (("N", "N", -0.8, 0.0), ("CA", "C", 0.0, 0.2),
                 ("C", "C", 0.8, 0.5), ("O", "O", 1.2, 0.8))
        lines = []
        coordinates = []
        serial = 0
        for chain, base_x, y in (("A", 2.0, 2.0), ("B", 18.0, 5.0)):
            for name, element, dx, z in atoms:
                serial += 1
                lines.append(pdb_line(
                    serial, name, "GLY", 1, base_x + dx, y, z,
                    element, chain=chain))
                coordinates.append([base_x + dx, y, z])
        lines.extend((
            "CONECT    1    2", "CONECT    2    3", "CONECT    3    4",
            "CONECT    5    6", "CONECT    6    7", "CONECT    7    8",
            "END"))
        with open(self.path("wrapped_chains.pdb"), "w") as handle:
            handle.write("\n".join(lines) + "\n")

        start = np.asarray(coordinates, dtype=np.float32)
        frames = []
        for chain_b_shift in (0.0, -20.0, 0.0, -20.0):
            positions = start.copy()
            positions[4:, 0] += chain_b_shift
            frames.append(positions)
        write_frames(
            self.path("wrapped_chains.pdb"), self.path("wrapped_chains.xtc"),
            frames, np.array([20, 20, 20, 90, 90, 90], dtype=np.float32))

        indices, labels, corrected = utils.topology_aware_ca_rmsf(
            self.path("wrapped_chains.pdb"), self.path("wrapped_chains.xtc"),
            self.path("wrapped_chains.pdb"))
        raw_universe = mda.Universe(
            self.path("wrapped_chains.pdb"), self.path("wrapped_chains.xtc"))
        try:
            _, _, coordinate_only = utils.backbone_aligned_ca_rmsf(raw_universe)
        finally:
            raw_universe.trajectory.close()

        self.assertEqual(indices.tolist(), [1, 2])
        self.assertEqual(labels.tolist(), ["A:GLY1", "B:GLY1"])
        np.testing.assert_allclose(corrected, 0.0, atol=1e-5)
        self.assertGreater(float(coordinate_only.max()), 1.0)

    def test_rmsf_rows_identify_duplicate_residue_numbers_across_chains(self):
        atoms = (("N", "N", 0.0), ("CA", "C", 1.4),
                 ("C", "C", 2.4), ("O", "O", 3.0))
        lines = []
        serial = 0
        for chain, y in (("A", 0.0), ("B", 5.0)):
            for name, element, x in atoms:
                serial += 1
                lines.append(pdb_line(
                    serial, name, "GLY", 1, x, y, 0.2 * serial,
                    element, chain=chain))
        with open(self.path("two_chains.pdb"), "w") as handle:
            handle.write("\n".join(lines) + "\nEND\n")
        write_trajectory(
            self.path("two_chains.pdb"), self.path("two_chains.xtc"),
            n_frames=3)

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                frame, _, status = module.on_analyze_rmsf(
                    self.working_directory_path,
                    "two_chains.pdb", "two_chains.xtc")
                self.assertIn("successfully", self.plain_text(status))
                self.assertEqual(frame["Residue Index"].tolist(), [1, 2])
                self.assertEqual(frame["Residue"].tolist(),
                                 ["A:GLY1", "B:GLY1"])
                self.assertTrue(frame["Residue"].is_unique)

    def test_rmsf_removes_rigid_translation_before_measuring_fluctuation(self):
        """The fixture moves every atom together, so aligned RMSF is zero."""
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                frame, _, status = self.analyse(module.on_analyze_rmsf)
                self.assertIn("successfully", self.plain_text(status))
                np.testing.assert_allclose(frame["Cα RMSF (Å)"], 0.0, atol=1e-5)

    def test_protein_rmsd_uses_the_whole_protein_after_the_backbone_fit(self):
        """Motion outside the fitted backbone must remain in the protein series."""
        structure = write_structure_pdb(
            self.path("sidechain.pdb"), n_residues=4, ions={"LIG": 1})
        rename_first_oxygen_as_sidechain(structure)

        universe = mda.Universe(structure)
        try:
            start = universe.atoms.positions.copy()
            sidechain = universe.select_atoms("protein and not backbone")
            self.assertEqual(sidechain.n_atoms, 1)
            sidechain_index = sidechain.indices[0]
        finally:
            universe.trajectory.close()

        frames = []
        for number in range(3):
            positions = start + number * np.array([2.0, -1.0, 0.5])
            positions[sidechain_index] += number * np.array([0.0, 4.0, 0.0])
            frames.append(positions)
        write_frames(structure, self.path("sidechain.xtc"), frames)

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                frame, _, status = module.on_analyze_rmsd(
                    self.working_directory_path, "sidechain.pdb", "sidechain.xtc")
                self.assertIn("successfully", self.plain_text(status))
                # The backbone itself only translated and has essentially zero
                # fit RMSD.  A whole-protein series must retain the moving CB.
                self.assertGreater(frame["Protein RMSD (Å)"].iloc[-1], 1.0)

    def test_ligand_rmsd_retains_motion_relative_to_the_fitted_protein(self):
        """An independent one-atom ligand fit would erase this displacement."""
        structure = self.path("system.pdb")
        universe = mda.Universe(structure)
        try:
            start = universe.atoms.positions.copy()
            ligand_indices = universe.select_atoms("resname LIG").indices
        finally:
            universe.trajectory.close()

        frames = []
        for number in range(3):
            positions = start + number * np.array([1.0, 2.0, -1.0])
            positions[ligand_indices] += number * np.array([3.0, 0.0, 0.0])
            frames.append(positions)
        write_frames(structure, self.path("moving_ligand.xtc"), frames)

        frame, _, status = complex_workflow.on_analyze_rmsd(
            self.working_directory_path, "system.pdb", "moving_ligand.xtc")
        self.assertIn("successfully", self.plain_text(status))
        self.assertLess(frame["Protein RMSD (Å)"].iloc[-1], 1e-4)
        self.assertGreater(frame["Ligand RMSD (Å)"].iloc[-1], 5.9)

    def test_minimum_distance_measures_the_closest_approach(self):
        frame = self.assertAnalysisSucceeded(
            self.analyse(complex_workflow.on_analyze_min_distance),
            ["Time (ns)", "Minimum distance (Å)"])

        self.assertEqual(len(frame), self.FRAMES)
        self.assertTrue((frame["Minimum distance (Å)"] > 0).all())

    def test_minimum_distance_helper_bounds_each_pairwise_allocation(self):
        first = np.array([[float(i), 0.0, 0.0] for i in range(7)])
        second = np.array([[20.0 + float(i), 0.0, 0.0] for i in range(5)])
        distance_array = complex_workflow.distances.distance_array

        with patch.object(complex_workflow.distances, "distance_array",
                          wraps=distance_array) as measured:
            result = complex_workflow._minimum_distance_in_chunks(
                first, second, None, max_pairs=6)

        self.assertEqual(result, 14.0)
        self.assertGreater(measured.call_count, 1)
        for call in measured.call_args_list:
            self.assertLessEqual(len(call.args[0]) * len(call.args[1]), 6)

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

    def test_com_distance_uses_topology_aware_gromacs_when_tpr_is_available(self):
        """The UI path must make molecules whole using TPR connectivity."""
        with open(self.path("md.tpr"), "wb") as handle:
            handle.write(b"test fixture")

        commands = []

        def fake_run(cmd, cwd=None, stdin_input=None):
            commands.append((cmd, cwd, stdin_input))
            output_name = cmd[cmd.index("-oall") + 1]
            with open(os.path.join(cwd, output_name), "w") as handle:
                handle.write("0.0 0.25\n1.0 0.40\n")
            return Mock(returncode=0)

        with patch.object(complex_workflow, "run_checked_command",
                          side_effect=fake_run):
            frame, figure, status = complex_workflow.on_analyze_com_distance(
                self.working_directory_path, "system.pdb", "traj.xtc", "md.tpr")

        self.assertIn("successfully", self.plain_text(status))
        self.assertIsInstance(figure, Figure)
        np.testing.assert_allclose(frame["Time (ns)"], [0.0, 1.0])
        np.testing.assert_allclose(
            frame["Center of mass distance (Å)"], [2.5, 4.0])

        self.assertEqual(len(commands), 1)
        cmd, cwd, stdin_input = commands[0]
        self.assertEqual(cmd[:2], ["gmx", "distance"])
        self.assertEqual(cmd[cmd.index("-s") + 1], "md.tpr")
        self.assertEqual(cmd[cmd.index("-f") + 1], "traj.xtc")
        self.assertEqual(
            cmd[cmd.index("-select") + 1],
            'com of group "Protein" plus com of resname LIG')
        self.assertEqual(cmd[cmd.index("-rmpbc") + 1], "yes")
        self.assertEqual(cmd[cmd.index("-pbc") + 1], "yes")
        self.assertEqual(cwd, os.path.abspath(self.working_directory_path))
        self.assertEqual(stdin_input, "")
        self.assertFalse(any(name.startswith(".com_distance_")
                             for name in os.listdir(self.working_directory_path)))

    def test_com_distance_uses_the_periodic_minimum_image(self):
        structure = write_structure_pdb(
            self.path("periodic.pdb"), n_residues=1, ions={"LIG": 1})
        universe = mda.Universe(structure)
        try:
            positions = universe.atoms.positions.copy()
            protein_indices = universe.select_atoms("protein").indices
            ligand_indices = universe.select_atoms("resname LIG").indices
        finally:
            universe.trajectory.close()

        positions[protein_indices] = np.array([1.0, 1.0, 1.0])
        positions[ligand_indices] = np.array([9.0, 1.0, 1.0])
        write_frames(structure, self.path("periodic.xtc"), [positions],
                     dimensions=np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]))

        frame, _, status = complex_workflow.on_analyze_com_distance(
            self.working_directory_path, "periodic.pdb", "periodic.xtc")
        self.assertIn("successfully", self.plain_text(status))
        self.assertAlmostEqual(frame["Center of mass distance (Å)"].iloc[0], 2.0, places=4)

    def test_trajectory_readers_close_even_when_plotting_fails(self):
        """An exception after iteration must not leave a large trajectory open."""
        handlers = (
            (workflow, "on_analyze_rmsd"),
            (workflow, "on_analyze_rmsf"),
            (complex_workflow, "on_analyze_rmsd"),
            (complex_workflow, "on_analyze_rmsf"),
            (complex_workflow, "on_analyze_com_distance"),
            (complex_workflow, "on_analyze_min_distance"),
        )
        real_universe = mda.Universe

        for module, handler_name in handlers:
            with self.subTest(module=module.__name__, handler=handler_name):
                opened = []

                def tracked_universe(*args, **kwargs):
                    universe = real_universe(*args, **kwargs)
                    universe.trajectory.close = Mock(wraps=universe.trajectory.close)
                    opened.append(universe)
                    return universe

                with patch.object(module.mda, "Universe", side_effect=tracked_universe), \
                        patch.object(module, "make_line_figure",
                                     side_effect=RuntimeError("plot failed")):
                    _, _, status = self.analyse(getattr(module, handler_name))

                self.assertIn("Error", self.plain_text(status))
                self.assertEqual(len(opened), 1)
                opened[0].trajectory.close.assert_called_once_with()

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
