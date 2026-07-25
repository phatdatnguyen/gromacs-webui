"""Tests for GRO parsing and protein-ligand structure/topology merging."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import utils

PROTEIN_GRO = textwrap.dedent("""\
    Protein
        3
        1ALA      N    1   1.000   2.000   3.000
        1ALA     CA    2   1.100   2.100   3.100
        1ALA      C    3   1.200   2.200   3.200
       5.00000   5.00000   5.00000
    """)

LIGAND_GRO = textwrap.dedent("""\
    Ligand
        2
        1LIG     C1    1   0.500   0.500   0.500
        1LIG     C2    2   0.600   0.600   0.600
       1.00000   1.00000   1.00000
    """)

PROTEIN_TOP = textwrap.dedent("""\
    ; Include forcefield parameters
    #include "amber99sb-ildn.ff/forcefield.itp"

    [ moleculetype ]
    Protein     3

    ; Include water topology
    #include "amber99sb-ildn.ff/tip3p.itp"

    [ system ]
    Protein in water

    [ molecules ]
    ; Compound        #mols
    Protein_chain_A     1
    SOL              1000
    """)


class ReadGromacsStructureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, content):
        path = os.path.join(self.directory.name, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def test_splits_title_count_atoms_and_box(self):
        title, natoms, atoms, box = utils.read_gromacs_structure_file(
            self.write("protein.gro", PROTEIN_GRO))
        self.assertEqual(title, "Protein")
        self.assertEqual(natoms, 3)
        self.assertEqual(len(atoms), 3)
        self.assertIn("5.00000", box)

    def test_merged_structure_sums_atoms_and_keeps_the_protein_box(self):
        protein = self.write("protein.gro", PROTEIN_GRO)
        ligand = self.write("ligand.gro", LIGAND_GRO)
        output = os.path.join(self.directory.name, "complex.gro")

        utils.merge_protein_ligand_structures(protein, ligand, output)

        title, natoms, atoms, box = utils.read_gromacs_structure_file(output)
        self.assertEqual(natoms, 5)
        self.assertEqual(len(atoms), 5)
        self.assertIn("LIG", "".join(atoms))
        self.assertIn("5.00000", box)          # protein box, not the ligand's 1 nm box
        self.assertNotIn("1.00000", box)


class MergeTopologyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.protein = os.path.join(self.directory.name, "topol.top")
        with open(self.protein, "w") as handle:
            handle.write(PROTEIN_TOP)
        self.ligand = os.path.join(self.directory.name, "ligand_GMX.itp")
        with open(self.ligand, "w") as handle:
            handle.write("[ moleculetype ]\nligand 3\n")
        self.output = os.path.join(self.directory.name, "complex.top")

    def merge(self):
        utils.merge_protein_ligand_topologies(self.protein, self.ligand, self.output)
        with open(self.output) as handle:
            return handle.read()

    def test_ligand_include_lands_after_the_forcefield_include(self):
        merged = self.merge()
        lines = [line.strip() for line in merged.splitlines() if line.strip()]
        forcefield = next(i for i, line in enumerate(lines) if "forcefield.itp" in line)
        include = next(i for i, line in enumerate(lines) if 'ligand_GMX.itp' in line)
        self.assertGreater(include, forcefield)
        # must come before the water topology so the atom types are defined in order
        water = next(i for i, line in enumerate(lines) if "tip3p.itp" in line)
        self.assertLess(include, water)

    def test_ligand_is_listed_in_the_molecules_section(self):
        merged = self.merge()
        molecules = merged.split("[ molecules ]")[1]
        self.assertRegex(molecules, r"ligand\s+1")

    def test_existing_content_is_preserved(self):
        merged = self.merge()
        self.assertIn("Protein_chain_A     1", merged)
        self.assertIn("SOL              1000", merged)

    def test_running_twice_does_not_duplicate_the_molecule_entry(self):
        self.merge()
        # merging the already-merged topology again must stay idempotent
        self.protein = self.output
        self.output = os.path.join(self.directory.name, "complex2.top")
        merged = self.merge()
        molecules = merged.split("[ molecules ]")[1]
        self.assertEqual(len(re_findall_ligand(molecules)), 1)


def re_findall_ligand(text: str) -> list[str]:
    import re
    return re.findall(r"^ligand\s+\d+", text, flags=re.MULTILINE)


if __name__ == "__main__":
    unittest.main()
