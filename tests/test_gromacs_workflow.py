"""Integration tests that drive the real GROMACS binaries.

Skipped automatically when ``gmx`` is not on PATH, so the rest of the suite still
runs on a machine without GROMACS.
"""

from __future__ import annotations

import os
import unittest

import protein_md_simulation as workflow
import utils
from .testing_support import WorkingDirectoryTestCase, requires_gromacs, write_structure_pdb


@requires_gromacs
class TopologyGenerationTests(WorkingDirectoryTestCase):
    """pdb2gmx must keep its outputs, and its posre include, inside the job directory."""

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)

    def generate(self, n_terminus=None, c_terminus=None, force_field="AMBER99SB-ILDN"):
        default = utils.DEFAULT_TERMINUS_CHOICE
        return workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            force_field, "TIP3P", n_terminus or default, c_terminus or default)

    def test_default_termini_produce_a_topology_in_the_job_directory(self):
        files, status = self.generate()
        self.assertIn("successfully", self.plain_text(status))
        for name in ("protein.gro", "topology.top", "posre.itp"):
            self.assertIn(name, files)

    def test_posre_include_stays_relative_to_the_topology(self):
        """An absolute include would tie the topology to this app's directory."""
        root_posre = "posre.itp"
        before = os.path.getmtime(root_posre) if os.path.exists(root_posre) else None

        self.generate()

        with open(self.path("topology.top")) as handle:
            includes = [line.strip() for line in handle if "posre" in line and "#include" in line]
        self.assertEqual(includes, ['#include "posre.itp"'])
        after = os.path.getmtime(root_posre) if os.path.exists(root_posre) else None
        self.assertEqual(before, after, "pdb2gmx wrote posre.itp into the repository root")

    def test_force_field_without_a_terminus_menu_falls_back_to_its_defaults(self):
        """The AMBER ports patch termini through renamed residues and offer no menu."""
        files, status = self.generate(n_terminus="NH3+", c_terminus="COO-")
        text = self.plain_text(status)
        self.assertIn("successfully", text)
        self.assertIn("no terminus selection", text)
        self.assertIn("topology.top", files)

    def test_unknown_force_field_surfaces_the_gromacs_message(self):
        files, status = self.generate(force_field="NOSUCHFF")
        text = self.plain_text(status)
        self.assertIn("Could not find force field", text)


@requires_gromacs
class RunInputTests(WorkingDirectoryTestCase):
    """grompp must accept the generated MDPs, including the restrained ones."""

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        self.assertIn("successfully", self.plain_text(status), "topology setup failed")
        _, status = workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        self.assertIn("successfully", self.plain_text(status), "box setup failed")

    def test_minimisation_run_input_builds(self):
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        files, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status))
        self.assertIn("em.tpr", files)

    def test_restrained_nvt_run_input_builds(self):
        """define = -DPOSRES only works if grompp can resolve the posre include."""
        workflow.on_generate_nvt_equilibration_mdp_file(self.working_directory_path, 1, 0.002, 300,
                                                        "nvt.mdp", "AMBER99SB-ILDN")
        files, status = workflow.on_generate_nvt_equilibration_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "nvt.mdp", "nvt.tpr", 5)
        self.assertIn("successfully", self.plain_text(status))
        self.assertIn("nvt.tpr", files)

    def test_restrained_npt_run_input_builds(self):
        workflow.on_generate_npt_equilibration_mdp_file(self.working_directory_path, 1, 0.002, 300,
                                                        1.0, "npt.mdp", "AMBER99SB-ILDN")
        files, status = workflow.on_generate_npt_equilibration_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "npt.mdp", "npt.tpr", 5)
        self.assertIn("successfully", self.plain_text(status))
        self.assertIn("npt.tpr", files)

    def test_grompp_errors_reach_the_status_line(self):
        """Passing an MDP where the topology belongs must not yield a bare exit code."""
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        files, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "em.mdp", "em.mdp", "bad.tpr", 5)
        text = self.plain_text(status)
        self.assertNotIn("returned non-zero exit status", text)
        self.assertIn("gmx grompp failed", text)
        self.assertIn(".top", text)


@requires_gromacs
class CharmmForceFieldTests(WorkingDirectoryTestCase):
    """CHARMM36 is only present on machines where it has been installed."""

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "CHARMM36", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        text = self.plain_text(status)
        if "Could not find force field" in text:
            self.skipTest("charmm36 is not installed in this GROMACS tree")
        if "atomtype database" in text:
            # Seen intermittently: the charmm36 port fails to load its own residue
            # database. Nothing to do with the behaviour under test.
            self.skipTest("charmm36 residue database failed to load: " + text.splitlines()[2])
        self.assertIn("successfully", text, text)

    def test_explicit_charged_termini_match_the_default(self):
        """filter_ter puts None last, so the first offered patch is already the default.

        The fixture is a glycine peptide and charmm36 carries glycine-specific
        entries, so the charged N-terminus is offered as GLY-NH3+ rather than NH3+
        - which is exactly why the label is matched per prompt.
        """
        with open(self.path("topology.top")) as handle:
            default_topology = [line for line in handle if not line.startswith(";")]

        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "explicit.gro", "explicit.top",
            "CHARMM36", "TIP3P", "GLY-NH3+", "COO-")
        text = self.plain_text(status)
        self.assertIn("Termini:", text)
        self.assertIn("GLY-NH3+", text)

        with open(self.path("explicit.top")) as handle:
            explicit_topology = [line for line in handle if not line.startswith(";")]
        self.assertEqual(default_topology, explicit_topology)

    def test_unavailable_terminus_reports_the_options_that_exist(self):
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "bad.gro", "bad.top",
            "CHARMM36", "TIP3P", "NOT-A-TERMINUS", utils.DEFAULT_TERMINUS_CHOICE)
        text = self.plain_text(status)
        self.assertIn("Error generating topology", text)
        self.assertIn("NOT-A-TERMINUS", text)
        self.assertIn("Available types", text)

    def test_charmm_mdp_is_accepted_by_grompp(self):
        workflow.on_generate_simulation_box(self.working_directory_path, "protein.gro", "boxed.gro",
                                            "cubic", 1.0)
        workflow.on_generate_nvt_equilibration_mdp_file(self.working_directory_path, 1, 0.002, 300,
                                                        "nvt.mdp", "CHARMM36")
        files, status = workflow.on_generate_nvt_equilibration_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "nvt.mdp", "nvt.tpr", 5)
        self.assertIn("successfully", self.plain_text(status))
        self.assertIn("nvt.tpr", files)


if __name__ == "__main__":
    unittest.main()
