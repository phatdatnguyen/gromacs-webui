"""Tests for pdb2gmx terminus menu parsing and answer resolution.

``resolve_terminus_selections`` is exercised against a stand-in for pdb2gmx, so
the subprocess plumbing (probe run, stdout parsing, answer ordering) is covered
without needing GROMACS installed.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import textwrap
import unittest

import utils

SINGLE_CHAIN_OUTPUT = """\
Opening force field file /usr/local/gromacs/share/gromacs/top/charmm36.ff/aminoacids.n.tdb
Select start terminus type for LEU-1
 0: NH3+
 1: NH2
 2: None
Start terminus LEU-1: NH3+
Select end terminus type for THR-244
 0: COO-
 1: COOH
 2: CT2
 3: None
End terminus THR-244: COO-
"""

PRO_START_OUTPUT = """\
Select start terminus type for PRO-1
 0: PRO-NH2+
 1: NH3+
 2: None
Select end terminus type for GLY-9
 0: COO-
 1: COOH
 2: None
"""


class MenuParsingTests(unittest.TestCase):
    def test_reads_both_menus_in_order(self):
        menus = utils._parse_terminus_menus(SINGLE_CHAIN_OUTPUT)
        self.assertEqual([menu["kind"] for menu in menus], ["start", "end"])
        self.assertEqual(menus[0]["residue"], "LEU-1")
        self.assertEqual(menus[0]["options"], [("0", "NH3+"), ("1", "NH2"), ("2", "None")])
        self.assertEqual(menus[1]["options"][1], ("1", "COOH"))

    def test_multiple_chains_produce_a_menu_each(self):
        menus = utils._parse_terminus_menus(SINGLE_CHAIN_OUTPUT + SINGLE_CHAIN_OUTPUT)
        self.assertEqual(len(menus), 4)
        self.assertEqual([menu["kind"] for menu in menus], ["start", "end", "start", "end"])

    def test_zwitterion_note_is_stripped_from_the_label(self):
        output = ("Select start terminus type for ALA-1\n"
                  " 0: ZWITTERION_NH3+ (only use with zwitterions containing exactly one residue)\n"
                  " 1: NH3+\n")
        options = utils._parse_terminus_menus(output)[0]["options"]
        self.assertEqual(options[0], ("0", "ZWITTERION_NH3+"))

    def test_numbered_lines_outside_a_menu_are_ignored(self):
        output = ("Reading protein.pdb...\n"
                  " 1: this is not a menu entry\n"
                  "Select start terminus type for ALA-1\n"
                  " 0: NH3+\n"
                  "\n"
                  " 9: after the menu ended\n")
        menus = utils._parse_terminus_menus(output)
        self.assertEqual(len(menus), 1)
        self.assertEqual(menus[0]["options"], [("0", "NH3+")])

    def test_no_menu_returns_empty(self):
        self.assertEqual(utils._parse_terminus_menus("nothing interesting here"), [])


class ResolveSelectionsTests(unittest.TestCase):
    """Drive the real function against a fake pdb2gmx."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def fake_pdb2gmx(self, printed_output: str, exit_code: int = 0) -> list[str]:
        script = os.path.join(self.directory.name, "fake_pdb2gmx.py")
        with open(script, "w") as handle:
            handle.write(textwrap.dedent(f"""
                import sys
                sys.stdout.write({printed_output!r})
                sys.stdout.flush()
                sys.stdin.read()
                sys.exit({exit_code})
                """))
        # -o and -p are rewritten by the probe, so they must be present
        return [sys.executable, script, "-o", "out.gro", "-p", "out.top"]

    def resolve(self, output, n_terminus, c_terminus, exit_code=0):
        return utils.resolve_terminus_selections(
            self.fake_pdb2gmx(output, exit_code), self.directory.name, n_terminus, c_terminus)

    def test_labels_map_to_the_menu_indices(self):
        answers, descriptions = self.resolve(SINGLE_CHAIN_OUTPUT, "NH2", "COOH")
        self.assertEqual(answers, "1\n1\n")
        self.assertEqual(descriptions, ["LEU-1 NH2", "THR-244 COOH"])

    def test_matching_is_per_prompt_not_a_fixed_index(self):
        """A PRO N-terminus lists PRO-NH2+ first, so NH3+ is index 1 there."""
        answers, descriptions = self.resolve(PRO_START_OUTPUT, "NH3+", "COO-")
        self.assertEqual(answers, "1\n0\n")
        self.assertEqual(descriptions, ["PRO-1 NH3+", "GLY-9 COO-"])

    def test_default_choice_takes_the_first_option(self):
        answers, descriptions = self.resolve(SINGLE_CHAIN_OUTPUT, utils.DEFAULT_TERMINUS_CHOICE, "COOH")
        self.assertEqual(answers, "0\n1\n")
        self.assertEqual(descriptions[0], "LEU-1 NH3+")

    def test_label_matching_is_case_insensitive(self):
        answers, _ = self.resolve(SINGLE_CHAIN_OUTPUT, "nh2", "coo-")
        self.assertEqual(answers, "1\n0\n")

    def test_unavailable_label_reports_the_real_options(self):
        with self.assertRaises(Exception) as caught:
            self.resolve(SINGLE_CHAIN_OUTPUT, "NOPE", "COO-")
        message = str(caught.exception)
        self.assertIn("NOPE", message)
        self.assertIn("NH3+, NH2, None", message)

    def test_probe_failure_is_reported(self):
        with self.assertRaises(Exception):
            self.resolve(SINGLE_CHAIN_OUTPUT, "NH2", "COOH", exit_code=1)

    def test_no_menu_reports_no_answers_so_the_caller_can_fall_back(self):
        """The AMBER ports offer no terminus menu; that is not an error."""
        answers, descriptions = self.resolve("no menus at all\n", "NH2", "COOH")
        self.assertIsNone(answers)
        self.assertEqual(descriptions, [])

    def test_probe_leftovers_are_cleaned_up(self):
        self.resolve(SINGLE_CHAIN_OUTPUT, "NH2", "COOH")
        leftovers = [name for name in os.listdir(self.directory.name)
                     if name.startswith(utils.PROBE_PDB2GMX_PREFIX)]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
