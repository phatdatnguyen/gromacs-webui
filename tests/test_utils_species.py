"""Tests for species detection and the representations built from it."""

from __future__ import annotations

import os
import re
import tempfile
import unittest

import MDAnalysis as mda

import utils
from .testing_support import write_structure_pdb


class IonElementTests(unittest.TestCase):
    def test_plain_element_names(self):
        for resname in ("NA", "CL", "K", "MG", "ZN", "FE", "CU"):
            with self.subTest(resname=resname):
                self.assertEqual(utils.get_ion_element(resname), resname)

    def test_charmm_charge_suffixes(self):
        """The whole *2P / *3P family must resolve, not just copper."""
        for resname, element in (("CU2P", "CU"), ("CU3P", "CU"), ("FE3P", "FE"), ("ZN2P", "ZN"),
                                 ("AG1P", "AG"), ("MN2P", "MN"), ("CO3P", "CO")):
            with self.subTest(resname=resname):
                self.assertEqual(utils.get_ion_element(resname), element)

    def test_force_field_aliases(self):
        for resname, element in (("SOD", "NA"), ("CLA", "CL"), ("POT", "K"), ("CAL", "CA"),
                                 ("CES", "CS"), ("LIT", "LI"), ("IOD", "I")):
            with self.subTest(resname=resname):
                self.assertEqual(utils.get_ion_element(resname), element)

    def test_charge_signs_and_case(self):
        self.assertEqual(utils.get_ion_element("na+"), "NA")
        self.assertEqual(utils.get_ion_element("Cl-"), "CL")

    def test_unrecognised_names_return_none(self):
        for resname in ("XY9Q", "LIG", "SOL", "", None):
            with self.subTest(resname=resname):
                self.assertIsNone(utils.get_ion_element(resname))


class StructureSpeciesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def species_of(self, **kwargs):
        path = os.path.join(self.directory.name, "structure.pdb")
        write_structure_pdb(path, **kwargs)
        return utils.get_structure_species(mda.Universe(path))

    def test_monatomic_residues_are_ions_and_get_element_colours(self):
        species = self.species_of(n_residues=3, ions={"NA": 2, "CL": 2, "CU2P": 1})
        by_name = {ion["resname"]: ion for ion in species["ions"]}
        self.assertEqual(set(by_name), {"NA", "CL", "CU2P"})
        self.assertEqual(by_name["CU2P"]["color"], utils.ELEMENT_COLORS["CU"])
        self.assertEqual(by_name["NA"]["count"], 2)
        self.assertTrue(all(ion["recognized"] for ion in species["ions"]))
        self.assertEqual(species["protein_residues"], 3)
        self.assertEqual(species["hetero"], [])

    def test_unrecognised_ion_is_flagged_and_coloured_magenta(self):
        species = self.species_of(n_residues=1, ions={"XY9Q": 1})
        ion = species["ions"][0]
        self.assertFalse(ion["recognized"])
        self.assertEqual(ion["color"], utils.UNKNOWN_SPECIES_COLOR)
        self.assertIn("unrecognised", utils.get_species_legend(species))

    def test_water_is_reported_separately_from_ions(self):
        species = self.species_of(n_residues=2, ions={"NA": 1}, n_waters=3)
        self.assertEqual(species["water"], ["SOL"])
        self.assertNotIn("SOL", [ion["resname"] for ion in species["ions"]])

    def test_ions_are_ordered_by_descending_count(self):
        species = self.species_of(n_residues=1, ions={"NA": 1, "CL": 4})
        self.assertEqual([ion["resname"] for ion in species["ions"]], ["CL", "NA"])

    def test_legend_lists_protein_ions_and_water(self):
        species = self.species_of(n_residues=2, ions={"NA": 2}, n_waters=1)
        legend = utils.get_species_legend(species)
        self.assertIn("protein 2 res", legend)
        self.assertIn("NA 2", legend)
        self.assertIn("SOL (water)", legend)


class RepresentationTests(unittest.TestCase):
    def make_species(self, protein_residues=40, ions=(), hetero=(), water=()):
        return {"protein_residues": protein_residues,
                "ions": [{"resname": name, "count": 1, "element": utils.get_ion_element(name),
                          "color": utils.ELEMENT_COLORS.get(utils.get_ion_element(name) or "",
                                                            utils.UNKNOWN_SPECIES_COLOR),
                          "recognized": utils.get_ion_element(name) is not None} for name in ions],
                "hetero": [{"resname": name, "count": 1, "atoms_per_residue": 20} for name in hetero],
                "water": list(water)}

    def test_one_sphere_per_ion_with_its_own_colour_plus_labels(self):
        javascript = utils.get_species_representations_js(self.make_species(ions=("NA", "CU2P")))
        self.assertIn('sele: "[NA]", color: "#AB5CF2"', javascript)
        self.assertIn('sele: "[CU2P]", color: "#C88033"', javascript)
        self.assertEqual(javascript.count('addRepresentation("spacefill"'), 2)
        self.assertIn('labelType: "resname"', javascript)
        # A fixed radius keeps a mis-guessed element from shrinking the sphere.
        self.assertIn('radiusType: "size"', javascript)

    def test_ligand_uses_ball_and_stick(self):
        javascript = utils.get_species_representations_js(self.make_species(hetero=("LIG",)))
        self.assertIn('addRepresentation("ball+stick", { sele: "[LIG]" })', javascript)

    def test_water_only_drawn_when_present(self):
        with_water = utils.get_species_representations_js(self.make_species(water=("SOL",)))
        without_water = utils.get_species_representations_js(self.make_species())
        self.assertIn('addRepresentation("line", { sele: "water"', with_water)
        self.assertNotIn("water", without_water)

    def test_short_peptides_fall_back_from_cartoon_to_licorice(self):
        """Cartoon draws nothing below ~20 residues, which looks like a broken viewer."""
        self.assertIn('addRepresentation("cartoon"',
                      utils.get_species_representations_js(self.make_species(protein_residues=240)))
        self.assertIn('addRepresentation("licorice", { sele: "protein"',
                      utils.get_species_representations_js(self.make_species(protein_residues=5)))

    def test_html_page_embeds_the_representations_and_no_placeholders(self):
        html = utils.get_trajectory_viewer_html("basename", 1234, 10, self.make_species(ions=("NA",)))
        self.assertIn("unpkg.com/ngl@2.4.0/dist/ngl.js", html)
        self.assertIn('sele: "[NA]"', html)
        self.assertIn('max="9"', html)                    # frame slider upper bound
        self.assertEqual(html.count('?ts=" + TS'), 2)     # structure and trajectory both cache-busted
        self.assertEqual(re.findall(r"__[A-Z_]+__", html), [], "template placeholder left unfilled")


if __name__ == "__main__":
    unittest.main()
