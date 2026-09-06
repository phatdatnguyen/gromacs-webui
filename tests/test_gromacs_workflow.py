"""Integration tests that drive the real GROMACS binaries.

Skipped automatically when ``gmx`` is not on PATH, so the rest of the suite still
runs on a machine without GROMACS.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
import unittest.mock

import numpy as np

import protein_md_simulation as workflow
import utils
from .testing_support import (WorkingDirectoryTestCase, final_result, requires_gromacs,
                              write_structure_pdb, write_trajectory)


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
        """A missing coordinate file must surface grompp's useful diagnostic."""
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        files, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "missing.gro", "topology.top", "em.mdp", "bad.tpr", 5)
        text = self.plain_text(status)
        self.assertNotIn("returned non-zero exit status", text)
        self.assertIn("gmx grompp failed", text)
        self.assertIn("missing.gro", text)


@requires_gromacs
class GromosRunInputTests(WorkingDirectoryTestCase):
    """The safe default must remain usable without hiding arbitrary warnings."""

    def test_known_twin_range_warning_is_surfaced_and_tpr_is_built(self):
        write_structure_pdb(self.path("protein.pdb"), n_residues=4)
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro",
            "topology.top", "GROMOS54A7", "SPC",
            utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        self.assertIn("successfully", self.plain_text(status))
        _, status = workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro",
            "cubic", 1.4, "GROMOS54A7")
        self.assertIn("successfully", self.plain_text(status))
        workflow.on_generate_energy_minimization_mdp_file(
            self.working_directory_path, "em.mdp", "GROMOS54A7")

        files, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top",
            "em.mdp", "em.tpr", 0, "GROMOS54A7")

        self.assertIn("em.tpr", files)
        self.assertIn("color:orange", status)
        text = self.plain_text(status)
        self.assertIn("historical twin-range", text)
        self.assertIn("No other grompp warning was bypassed", text)


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
        # NVT from pdb2gmx coordinates can explode before the hardware assignment
        # is observable.  Minimise first so this test measures CPU/GPU routing,
        # not the synthetic peptide fixture's starting strain.
        workflow.on_generate_energy_minimization_mdp_file(
            self.working_directory_path, "em.mdp", "AMBER99SB-ILDN")
        _, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top",
            "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "minimisation setup failed")
        _, status = workflow.on_run_energy_minimization(
            self.working_directory_path, "em.tpr", 1, 1, False)
        self.assertIn("completed successfully", self.plain_text(status),
                      "minimisation failed")
        workflow.on_generate_nvt_equilibration_mdp_file(
            self.working_directory_path, 0.02, 0.002, 300, "nvt.mdp", "AMBER99SB-ILDN")
        _, status = workflow.on_generate_nvt_equilibration_tpr_file(
            self.working_directory_path, "em.gro", "topology.top", "nvt.mdp", "nvt.tpr", 5)
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


class ConstraintDumpTests(AsyncRunMixin, WorkingDirectoryTestCase):
    """Crash diagnostics from a background process stay in its job directory.

    GROMACS writes ``step<n>b.pdb`` and ``step<n>c.pdb`` after a LINCS failure.
    A tiny child process writes the same hardcoded names so this remains a
    deterministic working-directory contract test instead of deliberately
    exploding a molecular system and occasionally spending minutes building an
    enormous pair list.
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
        with open(self.path("hot.tpr"), "wb") as handle:
            handle.write(b"process-launch fixture")

    def run_until_it_fails(self):
        """Start a failing child and wait for the watcher to clear its state."""
        real_popen = subprocess.Popen
        child_code = (
            "from pathlib import Path; "
            "Path('step0b.pdb').write_text('coordinates before\\n'); "
            "Path('step0c.pdb').write_text('coordinates after\\n'); "
            "raise SystemExit(1)"
        )

        def launch_in_requested_directory(_command, **kwargs):
            return real_popen(
                [sys.executable, "-c", child_code],
                cwd=kwargs.get("cwd"), text=kwargs.get("text", False),
                start_new_session=kwargs.get("start_new_session", False))

        with unittest.mock.patch.object(
                workflow.subprocess, "Popen",
                side_effect=launch_in_requested_directory):
            self.start_and_wait(
                workflow.on_run_nvt_equilibration,
                "hot.tpr", 1, 1, False, timeout=10)

    def test_the_dumps_land_in_the_job_directory_not_the_repository_root(self):
        self.run_until_it_fails()

        # Only crash dumps are asserted on, not "the root gained no files at all":
        # an unrelated tool dropping a log beside us is not this test's business,
        # and asserting on everything made it fail for reasons nothing to do with
        # mdrun. Checked before the skip below, since dumps here are the bug.
        self.assertEqual(self.dumps_in("."), [],
                         "mdrun scattered crash dumps into the repository root")

        dumps = self.dumps_in(self.working_directory_path)
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
        self.assertEqual(dumps, sorted(dumps, key=str.lower))


@requires_gromacs
class XvgRoundTripTests(WorkingDirectoryTestCase):
    """read_xvg against files gmx actually wrote, not against captured text.

    The unit tests in test_utils_xvg use fixtures, which only prove the parser
    matches what someone typed. This proves it matches GROMACS.
    """

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        _, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")
        # A rigid translation per frame: enough for the tools to have something to
        # read, and Rg/SASA are translation invariant so the values stay comparable.
        write_trajectory(self.path("boxed.gro"), self.path("traj.xtc"), n_frames=5)

    def test_a_real_gyrate_xvg_parses_into_named_columns(self):
        utils.run_checked_command(
            ["gmx", "gyrate", "-s", "em.tpr", "-f", "traj.xtc", "-o", "gyrate.xvg",
             "-sel", "protein"], cwd=self.working_directory_path)

        data = utils.read_xvg(self.path("gyrate.xvg"))
        self.assertEqual(len(data["frame"]), 5)
        self.assertIn("Rg", data["frame"].columns)
        self.assertEqual(data["xlabel"], "Time (ps)")
        # Four radii: total plus one per axis, all named from the file's own legends.
        self.assertEqual(len(data["frame"].columns), 5)
        self.assertFalse(data["frame"].isna().to_numpy().any())

    def test_a_real_sasa_xvg_parses_into_named_columns(self):
        utils.run_checked_command(
            ["gmx", "sasa", "-s", "em.tpr", "-f", "traj.xtc", "-o", "sasa.xvg",
             "-or", "resarea.xvg", "-surface", "protein", "-output", "protein"],
            cwd=self.working_directory_path)

        area = utils.read_xvg(self.path("sasa.xvg"))
        self.assertEqual(list(area["frame"].columns), ["Time (ps)", "Total", "Protein"])
        self.assertEqual(len(area["frame"]), 5)

        per_residue = utils.read_xvg(self.path("resarea.xvg"))
        self.assertEqual(per_residue["xlabel"], "Residue")
        self.assertEqual(per_residue["frame"]["Residue"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_group_numbers_are_read_from_a_real_gmx_menu(self):
        """Indices shift with force field and contents, so they must be looked up."""
        groups = utils.probe_gmx_groups(
            ["gmx", "covar", "-s", "em.tpr", "-f", "traj.xtc", "-o", "eigenval.xvg",
             "-v", "eigenvec.trr", "-av", "average.pdb", "-l", "covar.log"],
            cwd=self.working_directory_path)

        self.assertEqual(groups["System"], "0")
        self.assertEqual(groups["Protein"], "1")
        self.assertIn("C-alpha", groups)
        self.assertIn("Backbone", groups)
        self.assertEqual(utils.find_gmx_group_number(
            "\n".join(f"Group {number} ({name}) has 1 elements"
                      for name, number in groups.items()), "C-alpha"), groups["C-alpha"])


@requires_gromacs
class GmxAnalysisRunTests(WorkingDirectoryTestCase):
    """The real gmx sasa and gmx gyrate, driven with no interactive input."""

    FRAMES = 5

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        _, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")
        # Written from boxed.gro so the atom count matches the tpr exactly.
        write_trajectory(self.path("boxed.gro"), self.path("traj.xtc"), n_frames=self.FRAMES)

    def test_sasa_writes_both_xvg_files_and_returns_two_tables(self):
        files, area, area_figure, residue, residue_figure, status = final_result(
            workflow.on_analyze_sasa(
                self.working_directory_path, "em.tpr", "traj.xtc", "protein", "", 0.14,
                "sasa.xvg", "sasa_residue.xvg"))

        self.assertIn("successfully", self.plain_text(status), self.plain_text(status))
        self.assertIn("sasa.xvg", files)
        self.assertIn("sasa_residue.xvg", files)
        self.assertEqual(len(area), self.FRAMES)
        self.assertEqual(len(residue), 6)          # one row per residue
        self.assertIsNotNone(area_figure)
        self.assertIsNotNone(residue_figure)

    def test_gyrate_reports_the_total_radius_and_the_three_axes(self):
        files, frame, figure, status = final_result(workflow.on_analyze_gyrate(
            self.working_directory_path, "em.tpr", "traj.xtc", "protein", "mass", "gyrate.xvg"))

        self.assertIn("successfully", self.plain_text(status), self.plain_text(status))
        self.assertIn("gyrate.xvg", files)
        self.assertEqual(len(frame), self.FRAMES)
        self.assertEqual(len(frame.columns), 5)    # time, Rg, and one per axis
        self.assertIn("Rg", frame.columns)
        self.assertIsNotNone(figure)

    def test_the_probe_radius_reaches_gmx_and_changes_the_answer(self):
        """A bigger probe rolls over more crevices, so the area must differ."""
        _, small, _, _, _, _ = final_result(workflow.on_analyze_sasa(
            self.working_directory_path, "em.tpr", "traj.xtc", "protein", "", 0.10,
            "small.xvg", "small_residue.xvg"))
        _, large, _, _, _, _ = final_result(workflow.on_analyze_sasa(
            self.working_directory_path, "em.tpr", "traj.xtc", "protein", "", 0.25,
            "large.xvg", "large_residue.xvg"))

        self.assertNotAlmostEqual(small.iloc[0, 1], large.iloc[0, 1], places=3)

    def test_the_shipped_selection_defaults_are_accepted_by_gmx(self):
        """The defaults must parse, not just look reasonable.

        GROMACS reads a bare word as an index group name, and a group name may
        span several words, so "protein or resname LIG" is swallowed whole as one
        name and rejected. Only the explicit "group Protein or ..." form composes.
        Every default the UI ships is run through gmx here rather than trusted.
        """
        for selection in ("group Protein", "group Backbone", "group Protein or resname LIG",
                          "resname LIG", "protein"):
            with self.subTest(selection=selection):
                _, frame, _, status = final_result(workflow.on_analyze_gyrate(
                    self.working_directory_path, "em.tpr", "traj.xtc", selection, "mass",
                    "sel.xvg"))
                text = self.plain_text(status)
                self.assertNotIn("Invalid selection", text)
                self.assertNotIn("syntax error", text)

    def test_index_groups_are_read_from_a_real_make_ndx_listing(self):
        """gmx make_ndx prints "13 UNK : 74 atoms", a different shape from the
        "Group 13 ( UNK )" the analysis tools print. Parsed against the real
        thing, and it works on tpr versions newer than MDAnalysis can read."""
        groups = dict(utils.list_gmx_index_groups("em.tpr", self.working_directory_path))

        self.assertIn("System", groups)
        self.assertIn("Protein", groups)
        self.assertIn("C-alpha", groups)
        self.assertGreater(groups["System"], 0)
        self.assertEqual(groups["Protein"], groups["System"])   # vacuum fixture
        self.assertLess(groups["C-alpha"], groups["Protein"])

    def test_a_protein_only_system_suggests_no_ligand(self):
        self.assertEqual(
            utils.describe_selection_candidates("em.tpr", self.working_directory_path), "")

    def test_a_bare_word_combined_with_or_is_the_trap_it_looks_like(self):
        """Pins the mistake itself, so nobody reintroduces it as a "tidier" default."""
        _, frame, _, status = final_result(workflow.on_analyze_gyrate(
            self.working_directory_path, "em.tpr", "traj.xtc", "protein or resname LIG",
            "mass", "trap.xvg"))

        self.assertIsNone(frame)
        self.assertIn("Invalid selection", self.plain_text(status))

    def test_an_invalid_selection_fails_fast_instead_of_waiting_for_input(self):
        """The tools prompt interactively when a selection is missing; a wedged
        worker thread would never come back, so this must return promptly."""
        started = time.monotonic()
        files, frame, figure, status = final_result(workflow.on_analyze_gyrate(
            self.working_directory_path, "em.tpr", "traj.xtc", "nosuchkeyword", "mass",
            "bad.xvg"))
        elapsed = time.monotonic() - started

        self.assertIsNone(frame)
        self.assertIn("Error", self.plain_text(status))
        self.assertLess(elapsed, 30, "gmx gyrate appears to have blocked on stdin")
        self.assertNotIn("bad.xvg", files)

    def test_a_trajectory_that_does_not_match_the_tpr_is_reported(self):
        """The new .tpr dropdown makes an atom-count mismatch newly possible."""
        write_structure_pdb(self.path("other.pdb"), n_residues=2)
        write_trajectory(self.path("other.pdb"), self.path("other.xtc"), n_frames=3)

        _, frame, _, status = final_result(workflow.on_analyze_gyrate(
            self.working_directory_path, "em.tpr", "other.xtc", "protein", "mass", "mismatch.xvg"))

        self.assertIsNone(frame)
        self.assertIn("Error", self.plain_text(status))


@requires_gromacs
class PcaAndLandscapeTests(WorkingDirectoryTestCase):
    """The real gmx covar and anaeig, and the landscape built on their output.

    This is the test that proves the single-group index really does stop the two
    legacy tools from prompting. If that assumption ever breaks they block on
    stdin, so every call here is bounded by run_checked_command's closed stdin.
    """

    FRAMES = 30

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=8)
        workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "AMBER99SB-ILDN", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        workflow.on_generate_simulation_box(
            self.working_directory_path, "protein.gro", "boxed.gro", "cubic", 1.0)
        workflow.on_generate_energy_minimization_mdp_file(self.working_directory_path, "em.mdp",
                                                          "AMBER99SB-ILDN")
        _, status = workflow.on_generate_energy_minimization_tpr_file(
            self.working_directory_path, "boxed.gro", "topology.top", "em.mdp", "em.tpr", 5)
        self.assertIn("successfully", self.plain_text(status), "run input setup failed")
        # Internal motion, not just a rigid shift: the covariance matrix of a pure
        # translation is singular and PCA on it has nothing to decompose.
        write_trajectory(self.path("boxed.gro"), self.path("traj.xtc"),
                         n_frames=self.FRAMES, noise=0.4)

    def run_pca(self, first=1, second=2, selection="backbone"):
        return final_result(workflow.on_run_pca(
            self.working_directory_path, "em.tpr", "traj.xtc", selection, first, second,
            "pca_index.ndx", "pca_eigenvec.trr", "pca_eigenval.xvg", "pca_2dproj.xvg"))

    def test_pca_completes_without_waiting_for_a_group_selection(self):
        started = time.monotonic()
        files, eigenvalues, scree, projection, scatter, status = self.run_pca()
        elapsed = time.monotonic() - started

        self.assertIn("successfully", self.plain_text(status), self.plain_text(status))
        self.assertLess(elapsed, 60, "covar or anaeig appears to have blocked on stdin")
        for name in ("pca_index.ndx", "pca_eigenval.xvg", "pca_eigenvec.trr", "pca_2dproj.xvg"):
            self.assertIn(name, files)
        self.assertIsNotNone(scree)
        self.assertIsNotNone(scatter)

    def test_the_index_holds_exactly_one_group(self):
        """More than one and the legacy tools would ask which to use."""
        self.run_pca()
        with open(self.path("pca_index.ndx")) as handle:
            self.assertEqual(handle.read().count("["), 1)

    def test_the_projection_has_one_row_per_frame_and_two_components(self):
        _, _, _, projection, _, _ = self.run_pca()

        self.assertEqual(len(projection), self.FRAMES)
        self.assertEqual(len(projection.columns), 2)

    def test_eigenvalues_come_back_sorted_largest_first(self):
        _, eigenvalues, _, _, _, _ = self.run_pca()

        values = eigenvalues.iloc[:, 1].to_numpy()
        self.assertGreater(len(values), 1)
        self.assertTrue((values[:-1] >= values[1:]).all(), "eigenvalues are not descending")
        self.assertGreater(values[0], 0.0)

    def test_the_landscape_is_built_from_the_projection_on_disk(self):
        self.run_pca()

        frame, figure, status = workflow.on_analyze_free_energy_landscape(
            self.working_directory_path, "pca_2dproj.xvg", 300.0, 20)

        self.assertIn("successfully", self.plain_text(status), self.plain_text(status))
        self.assertEqual(len(frame), 20 * 20)
        self.assertEqual(frame["ΔG (kJ/mol)"].min(), 0.0)
        self.assertIsNotNone(figure)
        # Every bin the sample never reached is blank, not an infinite energy.
        self.assertFalse(np.isinf(frame["ΔG (kJ/mol)"].to_numpy()).any())

    def test_the_landscape_says_so_when_the_pca_has_not_been_run(self):
        frame, figure, status = workflow.on_analyze_free_energy_landscape(
            self.working_directory_path, "pca_2dproj.xvg", 300.0, 20)

        self.assertIsNone(frame)
        self.assertIn("Run the PCA first", self.plain_text(status))

    def test_an_impossible_selection_is_reported_rather_than_hanging(self):
        started = time.monotonic()
        _, eigenvalues, _, _, _, status = self.run_pca(selection="resname NOSUCHRESIDUE")

        self.assertLess(time.monotonic() - started, 30)
        self.assertIsNone(eigenvalues)
        self.assertIn("Error", self.plain_text(status))


@requires_gromacs
class CharmmForceFieldTests(WorkingDirectoryTestCase):
    """CHARMM36 is only present on machines where it has been installed."""

    def skip_if_charmm_is_flaking(self, status):
        """Skip when charmm36 failed to load its own database rather than when the
        behaviour under test broke.

        Measured at roughly a third of runs on GROMACS 2026.3, naming a different
        random residue each time (C321C/C6MNG, GR61/2300HG, AL2/POPI15 ...), so it
        is inside GROMACS, not here. setUp has always guarded its own pdb2gmx call;
        the tests below each run pdb2gmx again and were left unguarded, which is why
        the suite failed intermittently on whichever of them drew the short straw.
        """
        text = self.plain_text(status)
        if "Could not find force field" in text:
            self.skipTest("charmm36 is not installed in this GROMACS tree")
        # Two symptoms of the same instability: sometimes pdb2gmx reports a
        # missing atom type, sometimes it corrupts its heap and aborts (SIGABRT,
        # "free(): invalid pointer", exit status -6). Both are inside GROMACS.
        for symptom in ("atomtype database", "free(): invalid pointer",
                        "exit status -6", "double free"):
            if symptom in text:
                self.skipTest(f"charmm36 failed to load ({symptom})")
        if "file ffbonded.itp" in text and "Unknown " in text:
            self.skipTest("charmm36 produced an inconsistent ffbonded database")
        return text

    def setUp(self):
        super().setUp()
        write_structure_pdb(self.path("protein.pdb"), n_residues=6)
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "protein.gro", "topology.top",
            "CHARMM36", "TIP3P", utils.DEFAULT_TERMINUS_CHOICE, utils.DEFAULT_TERMINUS_CHOICE)
        text = self.skip_if_charmm_is_flaking(status)
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
        text = self.skip_if_charmm_is_flaking(status)
        self.assertIn("Termini:", text)
        self.assertIn("GLY-NH3+", text)

        with open(self.path("explicit.top")) as handle:
            explicit_topology = [line for line in handle if not line.startswith(";")]
        self.assertEqual(default_topology, explicit_topology)

    def test_unavailable_terminus_reports_the_options_that_exist(self):
        _, status = workflow.on_generate_protein_topology(
            self.working_directory_path, "protein.pdb", "bad.gro", "bad.top",
            "CHARMM36", "TIP3P", "NOT-A-TERMINUS", utils.DEFAULT_TERMINUS_CHOICE)
        text = self.skip_if_charmm_is_flaking(status)
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
        text = self.skip_if_charmm_is_flaking(status)
        self.assertIn("successfully", text)
        self.assertIn("nvt.tpr", files)


if __name__ == "__main__":
    unittest.main()
