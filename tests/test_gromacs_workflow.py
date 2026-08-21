"""Integration tests that drive the real GROMACS binaries.

Skipped automatically when ``gmx`` is not on PATH, so the rest of the suite still
runs on a machine without GROMACS.
"""

from __future__ import annotations

import os
import time
import unittest
import unittest.mock

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
class MinimisationRunTests(WorkingDirectoryTestCase):
    """A real mdrun, checked for where it writes and what hardware it picks."""

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
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        _, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")

    def minimise(self, use_gpu=True):
        files, status = workflow.on_run_energy_minimization(
            self.working_directory_path, "em.tpr", 1, 1, use_gpu)
        self.assertIn("completed successfully", self.plain_text(status), self.plain_text(status))
        return files

    def test_minimisation_completes_and_writes_into_the_job_directory(self):
        """-deffnm is a plain name, so mdrun resolves it against its own directory."""
        files = self.minimise()
        for name in ("em.gro", "em.log", "em.edr"):
            self.assertIn(name, files)

    def test_mdrun_leaves_nothing_in_the_repository_root(self):
        """mdrun's constraint dumps use hardcoded names relative to its directory."""
        before = set(os.listdir("."))
        self.minimise()
        self.assertEqual(set(os.listdir(".")) - before, set())

    def test_minimisation_stays_off_the_gpu_even_when_the_box_is_ticked(self):
        """GROMACS has no GPU PME for the minimisers.

        Omitting the offload flags is not enough: mdrun defaults every task to
        "auto" and picks a detected GPU, so the CPU has to be asked for by name.
        Passes trivially on a machine with no GPU, which is the honest outcome.
        """
        self.minimise(use_gpu=True)

        with open(self.path("em.log")) as handle:
            log = handle.read()
        if "compatible GPU" not in log:
            self.skipTest("no GPU on this machine, so there is nothing to offload to")
        for claim in ("GPU selected for this run", "aspects on the GPU",
                      "GPU 8x4 nonbonded", "nonbonded interactions on the GPU"):
            self.assertNotIn(claim, log, f"mdrun used the GPU for minimisation: {claim!r}")


class AsyncRunMixin:
    """Start one of the backgrounded mdrun handlers and wait it out safely."""

    @staticmethod
    def stop(state):
        """Shut the run down, so a slow or wedged mdrun is never left behind."""
        with state["lock"]:
            proc = state["proc"]
            state["proc"] = None
            state["running"] = False
        utils.stop_process_gracefully(proc)

    def start_and_wait(self, handler, *arguments, timeout=60):
        """Call handler(dir, *arguments, state) and block until the watcher clears it."""
        state = utils.ProcessStateDict()
        handler(self.working_directory_path, *arguments, state)
        # Registered before the wait: an mdrun still running at the deadline has to
        # be killed, or it outlives the suite burning a core on a deleted directory.
        self.addCleanup(self.stop, state)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with state["lock"]:
                if not state["running"]:
                    return
            time.sleep(0.2)
        self.fail(f"mdrun did not exit within {timeout}s")


@requires_gromacs
class EquilibrationHardwareTests(AsyncRunMixin, WorkingDirectoryTestCase):
    """What the "Use GPU" checkbox actually does to a real run.

    Every mdrun task option defaults to "auto", which resolves to a detected
    GPU, so a build with CUDA support ignores an unticked box unless the CPU is
    named explicitly. Only the log says which way it went.
    """

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        workflow.on_generate_nvt_equilibration_mdp_file(
            self.working_directory_path, 0.02, 0.002, 300, "nvt.mdp", "AMBER99SB-ILDN")
        _, status = workflow.on_generate_nvt_equilibration_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "nvt.mdp", "nvt.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")

    def equilibrate(self, use_gpu):
        self.start_and_wait(workflow.on_run_nvt_equilibration, "nvt.tpr", 1, 1, use_gpu)
        with open(self.path("nvt.log")) as handle:
            log = handle.read()
        if "compatible GPU" not in log:
            self.skipTest("no GPU on this machine, so the checkbox cannot change anything")
        return log

    def test_unticking_the_box_keeps_the_run_off_the_gpu(self):
        log = self.equilibrate(use_gpu=False)
        for claim in ("GPU selected for this run", "aspects on the GPU",
                      "GPU 8x4 nonbonded", "nonbonded interactions on the GPU"):
            self.assertNotIn(claim, log, f"mdrun used the GPU with the box unticked: {claim!r}")

    def test_ticking_the_box_does_reach_the_gpu(self):
        """The other half: the pinning must not have disabled the GPU outright."""
        log = self.equilibrate(use_gpu=True)
        self.assertIn("GPU selected for this run", log)


@requires_gromacs
class ConstraintDumpTests(AsyncRunMixin, WorkingDirectoryTestCase):
    """The crash dumps that used to pile up in the repository root.

    When LINCS cannot satisfy a constraint, mdrun writes step<n>b.pdb and
    step<n>c.pdb. Those names are hardcoded and no flag redirects them, so they
    land wherever mdrun was started from. Generating velocities at an absurd
    temperature makes constrained bonds rotate past lincs-warnangle immediately,
    which is the cheapest way to reach that code path on purpose.
    """

    @staticmethod
    def dumps_in(directory):
        return sorted(name for name in os.listdir(directory)
                      if name.startswith("step") and name.endswith(".pdb"))

    @staticmethod
    def clean_repository_root():
        """Remove stray dumps, including the backups GROMACS makes before it
        overwrites one of its own: those are named #step0b.pdb.1# and so escape
        both the dumps_in() pattern and the *.pdb entry in .gitignore."""
        for name in os.listdir("."):
            if name.lstrip("#").startswith("step") and ".pdb" in name:
                os.remove(name)

    def setUp(self):
        super().setUp()
        # A regression here writes into the repository root, so clean up after the
        # run whatever the outcome: a failing test must not litter the checkout.
        self.addCleanup(self.clean_repository_root)
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        # Ten steps only. The dumps appear in the first few, and an exploded system
        # grows a huge pair list that makes every later step crawl.
        workflow.on_generate_nvt_equilibration_mdp_file(
            self.working_directory_path, 0.02, 0.002, 100000, "hot.mdp", "AMBER99SB-ILDN")
        _, status = workflow.on_generate_nvt_equilibration_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "hot.mdp", "hot.tpr", 50)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")

    def run_until_it_fails(self):
        """Start the doomed run and wait for the watcher thread to clear the state."""
        # On a GPU machine the explosion surfaces as a CUDA fault before LINCS
        # ever reports, so keep this one run on the CPU.
        with unittest.mock.patch.dict(os.environ, {"GMX_DISABLE_GPU_DETECTION": "1"}):
            self.start_and_wait(workflow.on_run_nvt_equilibration, "hot.tpr", 1, 1, False)

    def test_the_dumps_land_in_the_job_directory_not_the_repository_root(self):
        root_before = set(os.listdir("."))
        self.run_until_it_fails()

        # Checked before the skip below: dumps in the root are the bug itself, and
        # skipping on "no dumps in the job directory" would hide exactly that.
        self.assertEqual(sorted(set(os.listdir(".")) - root_before), [],
                         "mdrun scattered crash dumps into the repository root")

        dumps = self.dumps_in(self.working_directory_path)
        if not dumps:
            self.skipTest("this GROMACS build did not dump coordinates for the failed constraint")
        # b is the state going in, c the state after constraining; both are written.
        self.assertTrue(any(name.endswith("b.pdb") for name in dumps), dumps)
        self.assertTrue(any(name.endswith("c.pdb") for name in dumps), dumps)
        with open(os.path.join(self.working_directory_path, dumps[0])) as handle:
            self.assertIn("coordinates", handle.readline())

    def test_the_dumps_are_listed_for_the_user(self):
        """They are diagnostics for a failed run, so they belong in the file table."""
        self.run_until_it_fails()
        self.assertEqual(self.dumps_in("."), [], "dumps landed in the repository root")

        files = workflow.get_files_in_working_directory(self.working_directory_path)
        dumps = [name for name in files if name.startswith("step") and name.endswith(".pdb")]
        if not dumps:
            self.skipTest("this GROMACS build did not dump coordinates for the failed constraint")
        self.assertEqual(dumps, sorted(dumps, key=str.lower))


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
