"""Tests for the trajectory and structure viewer data preparation.

These build a small structure and trajectory from scratch, so they verify the
frame arithmetic and the written files without needing GROMACS.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import MDAnalysis as mda

import utils
from .testing_support import (WorkingDirectoryTestCase, frames_of, write_structure_pdb,
                             write_trajectory)


class TrajectoryReductionTests(WorkingDirectoryTestCase):
    def setUp(self):
        super().setUp()
        self.structure = write_structure_pdb(self.path("system.pdb"), n_residues=4,
                                             ions={"NA": 2, "CU2P": 1}, n_waters=5)
        self.trajectory = write_trajectory(self.structure, self.path("traj.xtc"), n_frames=10)
        self.basename = "_unittest_traj"

    def reduce(self, selection="Protein", max_frames=200):
        return utils.write_trajectory_viewer_files(self.structure, self.trajectory,
                                                   selection, max_frames, self.basename)

    def written(self):
        return (os.path.join("static", self.basename + ".pdb"),
                os.path.join("static", self.basename + ".xtc"))

    def test_writes_a_matched_structure_and_trajectory(self):
        info = self.reduce()
        pdb, xtc = self.written()
        self.assertTrue(os.path.exists(pdb) and os.path.exists(xtc))

        universe = mda.Universe(pdb, xtc)
        self.addCleanup(universe.trajectory.close)
        self.assertEqual(universe.atoms.n_atoms, info["n_atoms"])
        self.assertEqual(len(universe.trajectory), info["frames"])

    def test_all_frames_kept_when_under_the_cap(self):
        info = self.reduce(max_frames=200)
        self.assertEqual(info["total_frames"], 10)
        self.assertEqual(info["frames"], 10)
        self.assertEqual(info["stride"], 1)

    def test_stride_is_computed_to_respect_the_cap(self):
        info = self.reduce(max_frames=4)
        self.assertEqual(info["stride"], 3)          # ceil(10 / 4)
        self.assertEqual(info["frames"], 4)          # frames 0, 3, 6, 9
        self.assertLessEqual(info["frames"], 4)

    def test_coordinate_budget_reduces_frames_for_large_selections(self):
        with mock.patch.object(
                utils, "MAX_TRAJECTORY_VIEWER_COORDINATES", 40):
            info = self.reduce(selection="All Atoms", max_frames=10)
        self.assertEqual(info["n_atoms"], 34)
        self.assertEqual(info["frames"], 1)
        self.assertEqual(info["stride"], 10)

    def test_partial_viewer_bundle_is_removed_after_writer_failure(self):
        class FailingWriter:
            def __init__(self, path, *_args, **_kwargs):
                self.path = path

            def __enter__(self):
                open(self.path, "wb").close()
                return self

            def write(self, _atoms):
                raise OSError("disk full")

            def __exit__(self, *_args):
                return False

        with mock.patch.object(utils.mda, "Writer", FailingWriter), \
                self.assertRaisesRegex(OSError, "disk full"):
            self.reduce()
        pdb, xtc = self.written()
        self.assertFalse(os.path.exists(pdb))
        self.assertFalse(os.path.exists(xtc))

    def test_frame_cap_is_strict_and_bounded_before_opening_the_trajectory(self):
        for value in (True, 0, -1, 1.5, 1001, float("nan"), float("inf"), "ten"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "integer from 1 to 1000"):
                self.reduce(max_frames=value)
        self.assertEqual(self.reduce(max_frames=10.0)["frames"], 10)

    def test_written_frames_are_distinct(self):
        """A viewer showing one repeated frame would look animated but be static."""
        self.reduce(max_frames=5)
        pdb, xtc = self.written()
        frames = frames_of(pdb, xtc)
        self.assertGreater(len(frames), 1)
        for earlier, later in zip(frames, frames[1:]):
            self.assertGreater(abs(later - earlier).max(), 1e-4)

    def test_protein_selection_drops_water_and_ions(self):
        info = self.reduce(selection="Protein")
        self.assertEqual(info["species"]["ions"], [])
        self.assertEqual(info["species"]["water"], [])
        self.assertEqual(info["n_atoms"], 4 * 4)     # four residues of four atoms

    def test_protein_ligand_ions_selection_keeps_ions_but_not_water(self):
        info = self.reduce(selection="Protein + Ligand + Ions")
        self.assertEqual({ion["resname"] for ion in info["species"]["ions"]}, {"NA", "CU2P"})
        self.assertEqual(info["species"]["water"], [])
        self.assertEqual(info["n_atoms"], 4 * 4 + 3)

    def test_all_atoms_selection_keeps_water(self):
        info = self.reduce(selection="All Atoms")
        self.assertEqual(info["species"]["water"], ["SOL"])
        self.assertEqual(info["n_atoms"], 4 * 4 + 3 + 5 * 3)

    def test_pdb_keeps_four_character_resnames(self):
        """NGL reads resname from columns 18-21; a clipped CU2P would not match."""
        self.reduce(selection="Protein + Ligand + Ions")
        pdb, _ = self.written()
        with open(pdb) as handle:
            resnames = {line[17:21].strip() for line in handle if line.startswith(("ATOM", "HETATM"))}
        self.assertIn("CU2P", resnames)

    def test_pdb_carries_element_symbols(self):
        self.reduce(selection="All Atoms")
        pdb, _ = self.written()
        with open(pdb) as handle:
            blank = [line for line in handle
                     if line.startswith(("ATOM", "HETATM")) and not line[76:78].strip()]
        self.assertEqual(blank, [])


class TrajectoryReductionErrorTests(WorkingDirectoryTestCase):
    """Deliberately broken inputs.

    MDAnalysis prints an ignored "XTCReader object has no attribute '_xdr'" during
    garbage collection when its reader fails to open a file. That noise comes from
    MDAnalysis' own __del__ and does not affect these assertions.
    """

    def test_mismatched_atom_counts_name_both_files(self):
        small = write_structure_pdb(self.path("small.pdb"), n_residues=1)
        big = write_structure_pdb(self.path("big.pdb"), n_residues=5)
        trajectory = write_trajectory(big, self.path("big.xtc"), n_frames=3)

        with self.assertRaises(Exception) as caught:
            utils.write_trajectory_viewer_files(small, trajectory, "Protein", 10, "_unittest_traj")
        message = str(caught.exception)
        self.assertIn("small.pdb", message)
        self.assertIn("big.xtc", message)
        self.assertIn("same run", message)

    def test_empty_trajectory_is_rejected_by_name(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        empty = self.path("empty.xtc")
        universe = mda.Universe(structure)
        with mda.Writer(empty, universe.atoms.n_atoms):
            pass

        with self.assertRaises(Exception) as caught:
            utils.write_trajectory_viewer_files(structure, empty, "Protein", 10, "_unittest_traj")
        self.assertIn("empty.xtc", str(caught.exception))

    def test_selection_matching_nothing_is_rejected(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        trajectory = write_trajectory(structure, self.path("traj.xtc"), n_frames=2)

        with self.assertRaises(Exception) as caught:
            utils.write_trajectory_viewer_files(structure, trajectory, "resname NOPE", 10, "_unittest_traj")
        self.assertIn("no atoms", str(caught.exception))


class StructureViewerPreparationTests(WorkingDirectoryTestCase):
    def test_gro_input_is_converted_and_species_reported(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=3, ions={"CU2P": 1})
        gro = self.path("system.gro")
        mda.Universe(structure).atoms.write(gro)

        display_path, species = utils.prepare_structure_viewer_file(gro, "static/_unittest_view.pdb")

        self.assertEqual(display_path, "static/_unittest_view.pdb")
        self.assertTrue(os.path.exists(display_path))
        self.assertEqual([ion["resname"] for ion in species["ions"]], ["CU2P"])
        with open(display_path) as handle:
            resnames = {line[17:21].strip() for line in handle if line.startswith(("ATOM", "HETATM"))}
        self.assertIn("CU2P", resnames)
        os.remove(display_path)

    def test_pdb_input_is_displayed_directly(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        display_path, species = utils.prepare_structure_viewer_file(structure, "static/_unittest_view.pdb")
        self.assertEqual(display_path, structure)
        self.assertFalse(os.path.exists("static/_unittest_view.pdb"))
        self.assertEqual(species["protein_residues"], 2)


if __name__ == "__main__":
    unittest.main()
