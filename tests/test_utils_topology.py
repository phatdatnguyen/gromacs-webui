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

LIGAND_ITP = textwrap.dedent("""\
    [ moleculetype ]
    ligand 3

    [ atoms ]
    ; nr type resnr residue atom cgnr charge mass
      1  c3     1    LIG   C1     1    0.0  12.011
      2  c3     1    LIG   C2     2    0.0  12.011
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

    def test_truncated_file_reports_a_clear_error(self):
        path = self.write("truncated.gro", "Protein\n2\n    1ALA      N    1   1.000   2.000   3.000\n")

        with self.assertRaisesRegex(ValueError, "truncated.*declares 2 atoms"):
            utils.read_gromacs_structure_file(path)

    def test_invalid_atom_coordinates_and_box_are_rejected(self):
        invalid_coordinate = PROTEIN_GRO.replace("   1.000", "     nan", 1)
        with self.assertRaisesRegex(ValueError, "non-finite coordinates"):
            utils.read_gromacs_structure_file(self.write("nan.gro", invalid_coordinate))

        invalid_box = PROTEIN_GRO.rsplit("\n", 2)[0] + "\n1.0 2.0\n"
        with self.assertRaisesRegex(ValueError, "three or nine finite box values"):
            utils.read_gromacs_structure_file(self.write("box.gro", invalid_box))

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

    def test_catastrophically_overlapping_atoms_are_rejected(self):
        protein = self.write("protein.gro", PROTEIN_GRO)
        ligand = self.write(
            "ligand.gro",
            LIGAND_GRO.replace("   0.500   0.500   0.500",
                               "   1.000   2.000   3.000", 1),
        )
        output = os.path.join(self.directory.name, "complex.gro")

        with self.assertRaisesRegex(ValueError, "catastrophically overlapping"):
            utils.merge_protein_ligand_structures(protein, ligand, output)
        self.assertFalse(os.path.exists(output))

    def test_unusually_far_ligand_is_merged_with_a_warning(self):
        protein = self.write("protein.gro", PROTEIN_GRO)
        ligand = self.write(
            "ligand.gro",
            LIGAND_GRO.replace("   0.500   0.500   0.500",
                               "  20.000  20.000  20.000", 1).replace(
                                   "   0.600   0.600   0.600",
                                   "  20.100  20.100  20.100", 1),
        )
        output = os.path.join(self.directory.name, "complex.gro")

        warnings = utils.merge_protein_ligand_structures(
            protein, ligand, output)

        self.assertTrue(os.path.isfile(output))
        self.assertEqual(len(warnings), 1)
        self.assertIn("unusually far", warnings[0])
        self.assertIn("same coordinate frame", warnings[0])


class LigandPairValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, content):
        path = os.path.join(self.directory.name, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def test_matching_ordered_gro_and_itp_are_accepted(self):
        gro = self.write("ligand_GMX.gro", LIGAND_GRO)
        itp = self.write("ligand_GMX.itp", LIGAND_ITP)

        self.assertIsNone(utils.validate_ligand_gro_itp_pair(gro, itp))

    def test_crossed_acpype_output_sets_are_rejected_even_with_same_atoms(self):
        gro = self.write("first_GMX.gro", LIGAND_GRO)
        itp = self.write("second_GMX.itp", LIGAND_ITP)

        with self.assertRaisesRegex(ValueError, "different ACPYPE output sets"):
            utils.validate_ligand_gro_itp_pair(gro, itp)

    def test_atom_count_and_order_mismatches_are_rejected(self):
        gro = self.write("ligand.gro", LIGAND_GRO)
        short_itp = self.write(
            "short.itp", LIGAND_ITP.replace(
                "  2  c3     1    LIG   C2     2    0.0  12.011\n", ""))
        with self.assertRaisesRegex(ValueError, "different atom counts"):
            utils.validate_ligand_gro_itp_pair(gro, short_itp)

        swapped_itp = self.write(
            "swapped.itp", LIGAND_ITP.replace("LIG   C1", "LIG   XX", 1).replace(
                "LIG   C2", "LIG   C1", 1).replace("LIG   XX", "LIG   C2", 1))
        with self.assertRaisesRegex(ValueError, "ordered atoms at position 1"):
            utils.validate_ligand_gro_itp_pair(gro, swapped_itp)

    def test_structure_merge_can_enforce_the_selected_topology_pair(self):
        protein = self.write("protein.gro", PROTEIN_GRO)
        gro = self.write("first_GMX.gro", LIGAND_GRO)
        itp = self.write("second_GMX.itp", LIGAND_ITP)
        output = os.path.join(self.directory.name, "complex.gro")

        with self.assertRaisesRegex(ValueError, "different ACPYPE output sets"):
            utils.merge_protein_ligand_structures(
                protein, gro, output, itp)
        self.assertFalse(os.path.exists(output))

    def test_topology_merge_can_enforce_the_selected_structure_pair(self):
        protein = self.write("protein.top", PROTEIN_TOP)
        gro = self.write("first_GMX.gro", LIGAND_GRO)
        itp = self.write("second_GMX.itp", LIGAND_ITP)
        output = os.path.join(self.directory.name, "complex.top")

        with self.assertRaisesRegex(ValueError, "different ACPYPE output sets"):
            utils.merge_protein_ligand_topologies(
                protein, itp, output, gro)
        self.assertFalse(os.path.exists(output))


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

    def test_molecules_entry_uses_the_itp_moleculetype_name(self):
        with open(self.ligand, "w") as handle:
            handle.write(textwrap.dedent("""\
                [ atomtypes ]
                c3  12.011  0.0  A  0.3  0.1

                [ moleculetype ]
                ; name       nrexcl
                Drug_X       3 ; an inline comment is valid here

                [ atoms ]
                """))

        merged = self.merge()
        molecules = merged.split("[ molecules ]")[1]
        self.assertRegex(molecules, r"(?m)^Drug_X\s+1$")
        self.assertNotRegex(molecules, r"(?m)^ligand\s+")

    def test_existing_content_is_preserved(self):
        merged = self.merge()
        self.assertIn("Protein_chain_A     1", merged)
        self.assertIn("SOL              1000", merged)

    def test_running_twice_does_not_duplicate_the_molecule_entry(self):
        first = self.merge()
        # merging the already-merged topology again must stay idempotent
        self.protein = self.output
        self.output = os.path.join(self.directory.name, "complex2.top")
        merged = self.merge()
        molecules = merged.split("[ molecules ]")[1]
        self.assertEqual(len(re_findall_ligand(molecules)), 1)
        self.assertEqual(merged.count('#include "ligand_GMX.itp"'), 1)
        self.assertEqual(merged, first)

    def test_remerge_with_a_different_ligand_replaces_the_managed_molecule(self):
        self.merge()
        new_ligand = os.path.join(self.directory.name, "drug_GMX.itp")
        with open(new_ligand, "w") as handle:
            handle.write("[ moleculetype ]\nDrug_X 3\n")

        self.protein = self.output
        self.ligand = new_ligand
        self.output = os.path.join(self.directory.name, "drug_complex.top")
        merged = self.merge()
        molecules = merged.split("[ molecules ]")[1]

        self.assertRegex(molecules, r"(?m)^Drug_X\s+1$")
        self.assertNotRegex(molecules, r"(?m)^ligand\s+")
        self.assertNotIn('#include "ligand_GMX.itp"', merged)
        self.assertEqual(merged.count('#include "drug_GMX.itp"'), 1)

    def test_remerge_upgrades_an_older_unmarked_molecule_row(self):
        older = self.merge()
        older = older.replace(utils._LIGAND_MOLECULE_BEGIN + "\n", "")
        older = older.replace(utils._LIGAND_MOLECULE_END + "\n", "")
        older = older.replace(utils._LIGAND_MOLECULE_NAME_PREFIX + "ligand\n", "")
        with open(self.output, "w") as handle:
            handle.write(older)

        new_ligand = os.path.join(self.directory.name, "replacement_GMX.itp")
        with open(new_ligand, "w") as handle:
            handle.write("[ moleculetype ]\nReplacement 3\n")
        self.protein = self.output
        self.ligand = new_ligand
        self.output = os.path.join(self.directory.name, "replacement.top")

        molecules = self.merge().split("[ molecules ]")[1]
        self.assertRegex(molecules, r"(?m)^Replacement\s+1$")
        self.assertNotRegex(molecules, r"(?m)^ligand\s+")

    def test_acpype_position_restraints_are_conditionally_included(self):
        with open(os.path.join(self.directory.name, "posre_ligand.itp"), "w") as handle:
            handle.write("[ position_restraints ]\n1 1 1000 1000 1000\n")

        merged = self.merge()

        restraint_block = textwrap.dedent("""\
            #ifdef POSRES_LIG
            #include "posre_ligand.itp"
            #endif
            """)
        self.assertIn(restraint_block, merged)
        self.assertLess(merged.index('#include "ligand_GMX.itp"'),
                        merged.index(restraint_block))

    def test_missing_position_restraint_file_does_not_create_a_broken_include(self):
        merged = self.merge()
        self.assertNotIn("POSRES_LIG", merged)
        self.assertNotIn("posre_ligand.itp", merged)

    def test_missing_forcefield_include_is_reported(self):
        with open(self.protein, "w") as handle:
            handle.write(PROTEIN_TOP.replace(
                '#include "amber99sb-ildn.ff/forcefield.itp"',
                '; force-field include was accidentally removed'))

        with self.assertRaisesRegex(ValueError, r"no forcefield\.itp include"):
            self.merge()

    def test_missing_moleculetype_name_is_reported(self):
        with open(self.ligand, "w") as handle:
            handle.write("[ atomtypes ]\nc3 12.011\n")

        with self.assertRaisesRegex(ValueError, r"molecule name.*\[ moleculetype \]"):
            self.merge()

    def test_multiple_ligand_molecule_types_are_rejected(self):
        with open(self.ligand, "w") as handle:
            handle.write(textwrap.dedent("""\
                [ moleculetype ]
                LIG 3
                [ atoms ]
                [ moleculetype ]
                COF 3
                """))

        with self.assertRaisesRegex(ValueError, "multiple molecule types"):
            self.merge()

    def test_unmanaged_ligand_molecule_row_is_not_silently_rewritten(self):
        with open(self.protein, "a") as handle:
            handle.write("ligand 4\n")

        with self.assertRaisesRegex(ValueError, "unmanaged molecule type 'ligand'"):
            self.merge()

    def test_existing_unmanaged_ligand_include_is_rejected(self):
        with open(self.protein) as handle:
            topology = handle.read()
        topology = topology.replace(
            '; Include water topology',
            '#include "ligand_GMX.itp"\n\n; Include water topology')
        with open(self.protein, "w") as handle:
            handle.write(topology)

        with self.assertRaisesRegex(ValueError, "already includes.*outside"):
            self.merge()

    def test_collision_with_inline_or_sibling_molecule_type_is_rejected(self):
        with open(self.protein) as handle:
            topology = handle.read()
        with open(self.protein, "w") as handle:
            handle.write(topology.replace("Protein     3", "ligand     3"))
        with self.assertRaisesRegex(ValueError, "already declares molecule type"):
            self.merge()

        with open(self.protein, "w") as handle:
            handle.write(PROTEIN_TOP.replace(
                '; Include water topology',
                '#include "protein_chain.itp"\n\n; Include water topology'))
        with open(os.path.join(self.directory.name, "protein_chain.itp"), "w") as handle:
            handle.write("[ moleculetype ]\nligand 3\n")
        with self.assertRaisesRegex(ValueError, "protein_chain.itp.*already declares"):
            self.merge()


LIGAND_PDB = textwrap.dedent("""\
    REMARK   1 an uploaded ligand
    HETATM    1  C1  UNK A 901      12.345  23.456  34.567  1.00  0.00           C
    HETATM    2  O1  UNK A 901      13.345  24.456  35.567  1.00  0.00           O
    TER       3      UNK A 901
    END
    """)


class RenamePdbResiduesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, content):
        path = os.path.join(self.directory.name, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def read(self, path):
        with open(path) as handle:
            return handle.read()

    def test_reports_and_replaces_the_old_residue_name(self):
        path = self.write("ligand.pdb", LIGAND_PDB)

        self.assertEqual(utils.rename_pdb_residues(path), ["UNK"])

        rewritten = self.read(path)
        self.assertNotIn("UNK", rewritten)
        self.assertEqual(rewritten.count("LIG"), 3)   # both atoms and the TER record

    def test_the_residue_name_stays_in_columns_18_to_20(self):
        """MDAnalysis reads the name by column, so a shifted record would break it."""
        path = self.write("ligand.pdb", LIGAND_PDB)
        utils.rename_pdb_residues(path)

        for line in self.read(path).splitlines():
            if line.startswith("HETATM"):
                self.assertEqual(line[17:20], "LIG")
                self.assertEqual(len(line), len(LIGAND_PDB.splitlines()[1]))

    def test_coordinates_and_other_records_are_untouched(self):
        path = self.write("ligand.pdb", LIGAND_PDB)
        utils.rename_pdb_residues(path)

        rewritten = self.read(path)
        self.assertIn("12.345  23.456  34.567", rewritten)
        self.assertIn("REMARK   1 an uploaded ligand", rewritten)
        self.assertTrue(rewritten.endswith("END\n"))

    def test_a_file_already_named_lig_is_left_alone(self):
        path = self.write("ligand.pdb", LIGAND_PDB.replace("UNK", "LIG"))
        before = os.stat(path).st_mtime_ns

        self.assertEqual(utils.rename_pdb_residues(path), [])
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_several_distinct_names_are_all_reported_once(self):
        path = self.write("ligand.pdb", LIGAND_PDB.replace("O1  UNK", "O1  MOL"))
        self.assertEqual(utils.rename_pdb_residues(path), ["UNK", "MOL"])


def re_findall_ligand(text: str) -> list[str]:
    import re
    return re.findall(r"^ligand\s+\d+", text, flags=re.MULTILINE)


if __name__ == "__main__":
    unittest.main()
