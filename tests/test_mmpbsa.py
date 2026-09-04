"""Tests for MM-PBSA: input generation, results parsing, and how it is launched.

A real run takes minutes to hours, so the command is checked with subprocess
mocked and the parser against a captured results file.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import unittest
import unittest.mock

import pandas as pd

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
import utils
from .testing_support import (WorkingDirectoryTestCase, write_structure_pdb, write_trajectory)

# Captured verbatim from a real gmx_MMPBSA 1.6.5 run of its own Protein_ligand
# single-trajectory example. Note what an invented fixture would have missed: the
# terms carry a Δ prefix, there are FIVE statistic columns rather than three, and
# a term name can contain a space ("Δ1-4 VDW").
RESULTS_DAT = textwrap.dedent("""\
    | Run on Fri Sep  4 15:07:52 2026
    | gmx_MMPBSA Version=v1.6.5 based on MMPBSA.py v.16.0

    ENERGY OF THE COMPLEX, RECEPTOR AND LIGAND
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------

    GENERALIZED BORN:

    Complex:
    Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
    -------------------------------------------------------------------------------
    BOND                    673.31          1.14      18.34         0.36       5.80
    VDWAALS                -732.66          1.18      12.83         0.37       4.06

    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    Receptor:
    Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
    -------------------------------------------------------------------------------
    BOND                    650.10          1.10      18.00         0.35       5.70
    VDWAALS                -709.09          1.15      12.50         0.36       3.95

    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    Ligand:
    Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
    -------------------------------------------------------------------------------
    BOND                     23.21          0.30       1.20         0.09       0.38

    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    Delta (Complex - Receptor - Ligand):
    Energy Component       Average     SD(Prop.)         SD   SEM(Prop.)        SEM
    -------------------------------------------------------------------------------
    ΔBOND                     0.00          1.14       0.00         0.36       0.00
    ΔANGLE                   -0.00          3.90       0.00         1.23       0.00
    ΔDIHED                   -0.00          1.67       0.00         0.53       0.00
    ΔVDWAALS                -23.57          1.18       2.83         0.37       0.90
    ΔEEL                     -7.11          0.77       3.19         0.24       1.01
    Δ1-4 VDW                 -0.00          0.57       0.00         0.18       0.00
    Δ1-4 EEL                  0.00          0.77       0.00         0.24       0.00
    ΔEGB                     17.85          0.14       3.04         0.04       0.96
    ΔESURF                   -2.19          0.02       0.06         0.01       0.02

    ΔGGAS                   -30.68          1.41       4.05         0.45       1.28
    ΔGSOLV                   15.67          0.14       3.04         0.05       0.96

    ΔTOTAL                  -15.02          1.42       1.69         0.45       0.53
    """)


class MmpbsaInputFileTests(WorkingDirectoryTestCase):
    def generate(self, methods=("MM-GBSA",), **kwargs):
        defaults = dict(start_frame=1, end_frame=0, interval=1,
                        salt_concentration=0.15, temperature=300.0)
        defaults.update(kwargs)
        return complex_workflow.on_generate_mmpbsa_input_file(
            self.working_directory_path, "mmpbsa.in", defaults["start_frame"],
            defaults["end_frame"], defaults["interval"], defaults["salt_concentration"],
            defaults["temperature"], list(methods), True, 2, "within 6")

    def read(self):
        with open(self.path("mmpbsa.in")) as handle:
            return handle.read()

    def test_gbsa_only_writes_a_gb_namelist(self):
        files, status = self.generate(methods=("MM-GBSA",))

        self.assertIn("mmpbsa.in", files)
        self.assertIn("successfully", self.plain_text(status))
        content = self.read()
        self.assertIn("&general", content)
        self.assertIn("&gb", content)
        self.assertNotIn("&pb", content)

    def test_selecting_both_methods_writes_both_namelists(self):
        self.generate(methods=("MM-GBSA", "MM-PBSA"))
        content = self.read()
        self.assertIn("&gb", content)
        self.assertIn("&pb", content)

    def test_selecting_no_method_is_refused(self):
        _, status = self.generate(methods=())
        self.assertIn("at least one", self.plain_text(status))

    def test_the_frame_range_and_conditions_reach_the_file(self):
        self.generate(start_frame=5, end_frame=250, interval=3,
                      salt_concentration=0.2, temperature=310.0)
        content = self.read()
        self.assertIn("startframe        = 5", content)
        self.assertIn("endframe          = 250", content)
        self.assertIn("interval          = 3", content)
        self.assertIn("temperature       = 310.0", content)
        self.assertIn("saltcon           = 0.2", content)

    def test_the_last_frame_is_left_open_when_the_end_is_zero(self):
        """gmx_MMPBSA wants a real frame number, so 0 means omit the key."""
        self.generate(end_frame=0)
        self.assertNotIn("endframe", self.read())


class MmpbsaFrameRangeTests(WorkingDirectoryTestCase):
    """The frame range is typed, so it has to be parsed and checked."""

    def generate(self, start="1", end="0", interval=100):
        return complex_workflow.on_generate_mmpbsa_input_file(
            self.working_directory_path, "mmpbsa.in", start, end, interval,
            0.15, 300.0, ["MM-GBSA"], True, 2, "within 6")

    def read(self):
        with open(self.path("mmpbsa.in")) as handle:
            return handle.read()

    def test_a_typed_range_reaches_the_input_file(self):
        """Typed, not dragged: a production trajectory holds more frames than a
        slider range would guess."""
        _, status = self.generate(start="2500", end="7500", interval=25)

        self.assertIn("successfully", self.plain_text(status))
        content = self.read()
        self.assertIn("startframe        = 2500", content)
        self.assertIn("endframe          = 7500", content)
        self.assertIn("interval          = 25", content)

    def test_zero_means_run_to_the_end(self):
        self.generate(end="0")
        self.assertNotIn("endframe", self.read())

    def test_surrounding_spaces_are_tolerated(self):
        self.generate(start="  10  ", end=" 20 ")
        self.assertIn("startframe        = 10", self.read())

    def test_something_that_is_not_a_number_names_the_field(self):
        for start, field in (("abc", "Start Frame"), ("", "Start Frame")):
            with self.subTest(value=start):
                _, status = self.generate(start=start)
                text = self.plain_text(status)
                self.assertIn(field, text)
                self.assertIn("whole number", text)

    def test_a_start_frame_below_one_is_refused(self):
        _, status = self.generate(start="0")
        self.assertIn("Start Frame must be 1 or greater", self.plain_text(status))

    def test_an_end_before_the_start_is_refused(self):
        _, status = self.generate(start="500", end="100")
        text = self.plain_text(status)
        self.assertIn("before Start Frame", text)
        self.assertIn("Use 0", text)


class MmpbsaResultsParsingTests(WorkingDirectoryTestCase):
    def parse(self, content=RESULTS_DAT):
        with open(self.path("FINAL_RESULTS_MMPBSA.dat"), "w") as handle:
            handle.write(content)
        return utils.parse_mmpbsa_results(self.path("FINAL_RESULTS_MMPBSA.dat"))

    def test_only_the_delta_section_is_read(self):
        """Complex, Receptor and Ligand repeat the same term names above it."""
        frame = self.parse()

        self.assertEqual(len(frame), 12)
        self.assertTrue(all(term.startswith("Δ") for term in frame["Term"]),
                        frame["Term"].tolist())
        # BOND appears in every section; only the delta one has the Δ prefix.
        self.assertNotIn("BOND", frame["Term"].tolist())

    def test_a_term_name_containing_a_space_is_kept_whole(self):
        """Splitting on whitespace naively would turn "Δ1-4 VDW" into two fields."""
        frame = self.parse().set_index("Term")

        self.assertIn("Δ1-4 VDW", frame.index)
        self.assertAlmostEqual(frame.loc["Δ1-4 VDW", "SD(Prop.)"], 0.57)

    def test_all_five_statistics_are_captured(self):
        frame = self.parse().set_index("Term")

        self.assertEqual(list(frame.columns), list(utils.MMPBSA_STATISTIC_COLUMNS))
        row = frame.loc["ΔVDWAALS"]
        self.assertAlmostEqual(row["Average (kcal/mol)"], -23.57)
        self.assertAlmostEqual(row["SD(Prop.)"], 1.18)
        self.assertAlmostEqual(row["SD"], 2.83)
        self.assertAlmostEqual(row["SEM(Prop.)"], 0.37)
        self.assertAlmostEqual(row["SEM"], 0.90)

    def test_the_binding_energy_is_the_delta_total(self):
        frame = self.parse().set_index("Term")

        self.assertAlmostEqual(frame.loc["ΔTOTAL", "Average (kcal/mol)"], -15.02)
        self.assertAlmostEqual(frame.loc["ΔTOTAL", "SD"], 1.69)

    def test_the_column_header_and_rules_are_not_read_as_terms(self):
        frame = self.parse()
        self.assertNotIn("Energy Component", frame["Term"].tolist())
        self.assertFalse(any(term.startswith("---") for term in frame["Term"]))

    def test_a_file_with_no_decomposition_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as caught:
            self.parse("gmx_MMPBSA crashed before writing anything useful.\n")
        self.assertIn("FINAL_RESULTS_MMPBSA.dat", str(caught.exception))

    def test_the_bar_chart_labels_every_term(self):
        figure = utils.make_bar_figure(self.parse(), "Term", "Average (kcal/mol)", "SD")
        axes = figure.axes[0]
        self.assertEqual(len(axes.patches), 12)


class RealMmpbsaOutputTests(WorkingDirectoryTestCase):
    """Parse a file gmx_MMPBSA actually wrote, not one we typed.

    Produced by `gmx_MMPBSA_test -t 3` (its own protein-ligand example). Skipped
    unless that output is present, since generating it needs the tool installed
    and takes minutes; the fixture above is a verbatim copy of the same file.
    """

    REAL_RESULTS = ("/tmp/mmtest/gmx_MMPBSA_test/examples/Protein_ligand/ST/"
                    "FINAL_RESULTS_MMPBSA.dat")

    def setUp(self):
        super().setUp()
        if not os.path.exists(self.REAL_RESULTS):
            self.skipTest("no gmx_MMPBSA example output on this machine")

    def test_the_real_file_parses_the_same_way_as_the_fixture(self):
        real = utils.parse_mmpbsa_results(self.REAL_RESULTS)

        with open(self.path("fixture.dat"), "w") as handle:
            handle.write(RESULTS_DAT)
        fixture = utils.parse_mmpbsa_results(self.path("fixture.dat"))

        self.assertEqual(real["Term"].tolist(), fixture["Term"].tolist())
        self.assertEqual(list(real.columns), list(fixture.columns))
        self.assertAlmostEqual(real.set_index("Term").loc["ΔTOTAL", "Average (kcal/mol)"],
                               fixture.set_index("Term").loc["ΔTOTAL", "Average (kcal/mol)"])


# Both captured verbatim from a real gmx_MMPBSA 1.6.3 decomposition run. Each
# file holds several tables separated by titles, so a plain read_csv would take
# the complex's numbers when the delta's are wanted.
PER_FRAME_CSV = textwrap.dedent("""\
    GENERALIZED BORN:
    Complex Energy Terms
    Frame #,BOND,ANGLE,DIHED,VDWAALS,EEL,1-4 VDW,1-4 EEL,EGB,ESURF,GGAS,GSOLV,TOTAL
    1,1531.71,3833.12,4955.97,-4406.08,-36258.07,1829.38,18874.82,-4107.88,108.2,-9639.14,-3999.68,-13638.82
    2,1666.3,3824.99,4986.1,-4249.48,-36139.65,1804.9,18770.14,-4217.01,120.32,-9336.7,-4096.69,-13433.39

    Delta Energy Terms
    Frame #,BOND,ANGLE,DIHED,VDWAALS,EEL,1-4 VDW,1-4 EEL,EGB,ESURF,GGAS,GSOLV,TOTAL
    1,-0.0,-0.0,-0.0,-42.09,-14.69,-0.0,0.0,32.98,-5.15,-56.78,27.84,-28.94
    2,0.0,0.0,0.0,-40.84,-7.88,-0.0,0.0,26.73,-4.86,-48.72,21.87,-26.85
    3,-0.0,0.0,0.0,-39.27,-12.77,-0.0,-0.0,30.63,-4.77,-52.04,25.86,-26.19
    """)

DECOMP_CSV = textwrap.dedent("""\
    Complex:
    Total Decomposition Contribution (TDC)
    Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL
    1,R:A:TRP:59,0.0,-99.0,-99.0,-99.0,-99.0,-99.0

    DELTAS:
    Total Decomposition Contribution (TDC)
    Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL
    1,R:A:TRP:59,0.0,-3.51,-0.44,1.46,-0.43,-2.92
    1,R:A:ARG:195,0.0,-0.73,-2.35,-0.06,-0.04,-3.17
    1,L:B:UNK:497,0.0,-21.04,-7.34,8.18,-3.27,-23.47
    2,R:A:TRP:59,0.0,-2.96,-0.38,1.16,-0.39,-2.58
    2,R:A:ARG:195,0.0,-0.53,-2.15,-0.06,-0.04,-2.77
    2,L:B:UNK:497,0.0,-19.04,-6.34,7.18,-3.07,-21.27

    Sidechain Decomposition Contribution (SDC)
    Frame #,Residue,Internal,van der Waals,Electrostatic,Polar Solvation,Non-Polar Solv.,TOTAL
    1,R:A:TRP:59,0.0,-1.0,-1.0,-1.0,-1.0,-1.0
    """)


class MmpbsaPerFrameTests(WorkingDirectoryTestCase):
    """The -eo file: every energy term for every frame."""

    def parse(self, content=PER_FRAME_CSV):
        with open(self.path("per_frame.csv"), "w") as handle:
            handle.write(content)
        return utils.parse_mmpbsa_per_frame(self.path("per_frame.csv"))

    def test_the_delta_table_is_read_not_the_complex_one(self):
        """Both tables carry the same column names; only the delta is the binding
        energy. Taking the first would report about -13600 instead of -27."""
        frame = self.parse()

        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(frame["TOTAL"].iloc[0], -28.94)
        self.assertNotIn(-13638.82, frame["TOTAL"].tolist())

    def test_the_columns_are_numeric_apart_from_nothing(self):
        frame = self.parse()
        self.assertAlmostEqual(frame["TOTAL"].mean(), (-28.94 - 26.85 - 26.19) / 3)
        self.assertAlmostEqual(frame["VDWAALS"].iloc[1], -40.84)

    def test_a_file_without_the_delta_table_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("GENERALIZED BORN:\nComplex Energy Terms\nFrame #,TOTAL\n1,5.0\n")

    def test_the_histogram_marks_the_mean(self):
        frame = self.parse()
        figure = utils.make_histogram_figure(frame["TOTAL"], bins=5,
                                             xlabel="ΔG binding (kcal/mol)")
        axes = figure.axes[0]
        self.assertEqual(axes.get_ylabel(), "Frames")
        self.assertAlmostEqual(axes.lines[0].get_xdata()[0], frame["TOTAL"].mean())


class MmpbsaDecompositionTests(WorkingDirectoryTestCase):
    """The -deo file: each printed residue's contribution, per frame."""

    def parse(self, content=DECOMP_CSV):
        with open(self.path("decomp.csv"), "w") as handle:
            handle.write(content)
        return utils.parse_mmpbsa_decomposition(self.path("decomp.csv"))

    def test_the_deltas_section_is_read_not_the_complex_one(self):
        """The complex's table repeats the same residues and column names."""
        frame = self.parse()

        self.assertEqual(len(frame), 3)
        self.assertNotIn(-99.0, frame["TOTAL"].tolist())

    def test_contributions_are_averaged_over_the_frames(self):
        frame = self.parse().set_index("Residue")

        self.assertAlmostEqual(frame.loc["R:A:TRP:59", "TOTAL"], (-2.92 + -2.58) / 2)
        self.assertAlmostEqual(frame.loc["L:B:UNK:497", "van der Waals"], (-21.04 + -19.04) / 2)

    def test_the_spread_across_frames_is_kept(self):
        """A residue whose contribution swings between frames is worth spotting."""
        frame = self.parse().set_index("Residue")
        self.assertIn("TOTAL SD", frame.columns)
        self.assertGreater(frame.loc["L:B:UNK:497", "TOTAL SD"], 0)

    def test_the_strongest_contribution_sorts_first(self):
        frame = self.parse()
        self.assertEqual(frame["Residue"].iloc[0], "L:B:UNK:497")

    def test_only_the_total_subsection_is_used(self):
        """Sidechain and backbone tables repeat every residue name."""
        frame = self.parse()
        self.assertEqual(sorted(frame["Residue"]),
                         ["L:B:UNK:497", "R:A:ARG:195", "R:A:TRP:59"])

    def test_a_file_without_a_deltas_section_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as caught:
            self.parse("Complex:\nTotal Decomposition Contribution (TDC)\n"
                       "Frame #,Residue,TOTAL\n1,R:A:TRP:59,-1.0\n")
        self.assertIn("DELTAS", str(caught.exception))


class MmpbsaDecompositionInputTests(WorkingDirectoryTestCase):
    """Without a &decomp namelist gmx_MMPBSA reports no residue contributions."""

    def generate(self, enabled=True, scheme=2, residues="within 6"):
        return complex_workflow.on_generate_mmpbsa_input_file(
            self.working_directory_path, "mmpbsa.in", "1", "0", 100, 0.15, 300.0,
            ["MM-GBSA"], enabled, scheme, residues)

    def read(self):
        with open(self.path("mmpbsa.in")) as handle:
            return handle.read()

    def test_decomposition_is_requested_by_default(self):
        self.generate()
        content = self.read()
        self.assertIn("&decomp", content)
        self.assertIn("idecomp           = 2", content)
        self.assertIn('print_res         = "within 6"', content)

    def test_the_scheme_and_residue_selection_come_from_the_ui(self):
        self.generate(scheme=1, residues="within 4")
        content = self.read()
        self.assertIn("idecomp           = 1", content)
        self.assertIn('print_res         = "within 4"', content)

    def test_it_can_be_turned_off(self):
        self.generate(enabled=False)
        self.assertNotIn("&decomp", self.read())

    def test_the_run_asks_for_the_decomposition_outputs(self):
        """-do and -deo are what produce the residue files at all."""
        for name in ("md.tpr", "md.xtc", "topol.top", "mmpbsa.in", "mmpbsa_index.ndx"):
            with open(self.path(name), "w") as handle:
                handle.write("x")
        state = utils.ProcessStateDict()
        with unittest.mock.patch.object(complex_workflow, "get_gmx_mmpbsa_executable",
                                        return_value="/opt/gmxMMPBSA/bin/gmx_MMPBSA"), \
                unittest.mock.patch.object(complex_workflow, "_build_mmpbsa_index"), \
                unittest.mock.patch.object(complex_workflow, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(complex_workflow, "threading"):
            fake_subprocess.DEVNULL = subprocess.DEVNULL
            fake_subprocess.STDOUT = subprocess.STDOUT
            complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", "group Protein", "resname LIG", 1, state)

        cmd = fake_subprocess.Popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-do") + 1], "FINAL_DECOMP_MMPBSA.dat")
        self.assertEqual(cmd[cmd.index("-deo") + 1], "FINAL_DECOMP_MMPBSA.csv")
        self.assertEqual(cmd[cmd.index("-eo") + 1], "FINAL_RESULTS_MMPBSA.csv")


class MmpbsaResidueColourTests(unittest.TestCase):
    """The ligand's own term dwarfs every residue, so it gets its own colour."""

    RESIDUES = ["L:B:LIG:245", "R:A:LEU:37", "R:A:GLU:44"]

    def test_the_ligand_is_coloured_apart_from_the_receptor(self):
        colours, legend = utils.mmpbsa_residue_colours(self.RESIDUES)

        self.assertEqual(colours[0], utils.MMPBSA_LIGAND_BAR_COLOUR)
        self.assertEqual(colours[1:], [utils.MMPBSA_RECEPTOR_BAR_COLOUR] * 2)
        self.assertNotEqual(utils.MMPBSA_LIGAND_BAR_COLOUR, utils.MMPBSA_RECEPTOR_BAR_COLOUR)

    def test_both_kinds_are_named_in_the_legend(self):
        _, legend = utils.mmpbsa_residue_colours(self.RESIDUES)
        self.assertEqual(sorted(legend.values()), ["Ligand", "Receptor residue"])

    def test_a_chart_with_no_ligand_row_does_not_claim_one(self):
        _, legend = utils.mmpbsa_residue_colours(["R:A:LEU:37", "R:A:GLU:44"])
        self.assertEqual(list(legend.values()), ["Receptor residue"])

    def test_the_colours_reach_the_bars(self):
        frame = pd.DataFrame({"Residue": self.RESIDUES, "TOTAL": [-13.6, -1.8, -1.4]})
        colours, legend = utils.mmpbsa_residue_colours(frame["Residue"])
        figure = utils.make_bar_figure(frame, "Residue", "TOTAL", colors=colours, legend=legend)

        axes = figure.axes[0]
        self.assertNotEqual(axes.patches[0].get_facecolor(), axes.patches[1].get_facecolor())
        self.assertIsNotNone(axes.get_legend())


class MmpbsaFrameTimeTests(WorkingDirectoryTestCase):
    """gmx_MMPBSA numbers its frames 1..N over the ones it selected."""

    def test_the_frame_selection_is_read_back_from_the_input_file(self):
        with open(self.path("mmpbsa.in"), "w") as handle:
            handle.write("&general\n  startframe        = 5,\n  interval          = 100,\n/\n")

        self.assertEqual(utils.read_mmpbsa_frame_selection(self.path("mmpbsa.in")), (5, 100))

    def test_an_input_file_without_them_falls_back_to_every_frame(self):
        with open(self.path("mmpbsa.in"), "w") as handle:
            handle.write("&general\n  temperature       = 300.0,\n/\n")

        self.assertEqual(utils.read_mmpbsa_frame_selection(self.path("mmpbsa.in")), (1, 1))

    def test_times_follow_the_start_and_interval_into_the_trajectory(self):
        """Frame 1 of the results is trajectory frame startframe-1, then every
        interval-th one after it - not frames 1, 2, 3."""
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        write_trajectory(structure, self.path("traj.xtc"), n_frames=20)

        every = utils.get_trajectory_frame_times_ns(structure, self.path("traj.xtc"), 1, 1, 20)
        strided = utils.get_trajectory_frame_times_ns(structure, self.path("traj.xtc"), 1, 5, 4)

        self.assertEqual(len(every), 20)
        self.assertEqual(len(strided), 4)
        self.assertEqual(strided, [every[0], every[5], every[10], every[15]])

    def test_asking_for_more_frames_than_exist_stops_at_the_end(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        write_trajectory(structure, self.path("traj.xtc"), n_frames=6)

        times = utils.get_trajectory_frame_times_ns(structure, self.path("traj.xtc"), 1, 1, 50)
        self.assertEqual(len(times), 6)


class MmpbsaAvailabilityTests(WorkingDirectoryTestCase):
    def absent(self):
        """Hide every place gmx_MMPBSA is looked for: the variable, the project's
        own environment, and PATH.

        _is_executable is patched rather than the environment path, so the hint
        still quotes the real documented location.
        """
        return unittest.mock.patch.multiple(
            utils,
            _is_executable=lambda path: False,
            shutil=unittest.mock.MagicMock(which=lambda name: None))

    def test_absence_is_reported_with_an_install_hint(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(utils.GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE, None)
            with self.absent():
                reason = utils.get_gmx_mmpbsa_unavailable_reason()
                self.assertFalse(utils.is_gmx_mmpbsa_available())

        self.assertIsNotNone(reason)
        self.assertIn("gmx_MMPBSA", reason)
        self.assertIn("conda create", reason)
        self.assertIn(utils.GMX_MMPBSA_ENVIRONMENT_PATH, reason)

    def test_the_project_environment_is_found_without_any_configuration(self):
        """The Readme tells you to build ./gmx-mmpbsa-env, so it must just work."""
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(utils.GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE, None)
            with unittest.mock.patch.object(utils, "_is_executable",
                                            side_effect=lambda path: "gmx-mmpbsa-env" in path):
                executable = utils.get_gmx_mmpbsa_executable()

        self.assertIsNotNone(executable)
        self.assertTrue(executable.endswith(os.path.join("gmx-mmpbsa-env", "bin", "gmx_MMPBSA")))
        self.assertTrue(os.path.isabs(executable), "the executable path must not be relative")

    def test_the_environment_variable_wins_over_path(self):
        with unittest.mock.patch.dict(
                os.environ, {utils.GMX_MMPBSA_EXECUTABLE_ENVIRONMENT_VARIABLE: "/no/such/binary"}):
            self.assertIsNone(utils.get_gmx_mmpbsa_executable())
            self.assertIn("gmx_MMPBSA", utils.get_gmx_mmpbsa_unavailable_reason())

    def test_the_run_refuses_to_start_when_it_is_not_installed(self):
        """Both helpers are patched: leaving the reason unpatched on a machine
        where gmx_MMPBSA *is* installed would surface the literal word "None"."""
        state = utils.ProcessStateDict()
        with unittest.mock.patch.object(complex_workflow, "get_gmx_mmpbsa_executable",
                                        return_value=None), \
                unittest.mock.patch.object(complex_workflow,
                                           "get_gmx_mmpbsa_unavailable_reason",
                                           return_value="MM-PBSA is disabled: gmx_MMPBSA "
                                                        "was not found."), \
                unittest.mock.patch.object(complex_workflow, "subprocess") as fake_subprocess:
            files, status, returned_state, button = complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", "protein", "resname LIG", 1, state)

        fake_subprocess.Popen.assert_not_called()
        self.assertIn("gmx_MMPBSA", self.plain_text(status))
        self.assertFalse(returned_state["running"])
        self.assertEqual(button["value"], "Start")

    def test_the_protein_only_tab_does_not_offer_mmpbsa(self):
        """A binding energy needs two partners, so it has no meaning there."""
        self.assertFalse(hasattr(workflow, "on_run_mmpbsa"))

    def test_a_disagreement_between_the_helpers_never_shows_the_word_none(self):
        """get_gmx_mmpbsa_unavailable_reason() returns None when the binary is
        found; if the two ever disagree the user must still get a real sentence."""
        state = utils.ProcessStateDict()
        with unittest.mock.patch.object(complex_workflow, "get_gmx_mmpbsa_executable",
                                        return_value=None), \
                unittest.mock.patch.object(complex_workflow,
                                           "get_gmx_mmpbsa_unavailable_reason",
                                           return_value=None), \
                unittest.mock.patch.object(complex_workflow, "subprocess"):
            _, status, _, _ = complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", "protein", "resname LIG", 1, state)

        text = self.plain_text(status)
        self.assertNotIn("None", text)
        self.assertIn("gmx_MMPBSA was not found", text)


class MmpbsaCommandTests(WorkingDirectoryTestCase):
    """What actually reaches Popen, without running the real thing."""

    def launch(self, processes=1, receptor="protein", ligand="resname LIG"):
        state = utils.ProcessStateDict()
        for name in ("md.tpr", "md.xtc", "topol.top", "mmpbsa.in", "mmpbsa_index.ndx"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        with unittest.mock.patch.object(complex_workflow, "get_gmx_mmpbsa_executable",
                                        return_value="/opt/gmxMMPBSA/bin/gmx_MMPBSA"), \
                unittest.mock.patch.object(complex_workflow, "_build_mmpbsa_index"), \
                unittest.mock.patch.object(complex_workflow, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(complex_workflow, "threading"):
            fake_subprocess.DEVNULL = subprocess.DEVNULL
            result = complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", receptor, ligand, processes, state)
        return fake_subprocess.Popen.call_args, result

    def test_the_command_names_every_input_and_both_groups(self):
        call, _ = self.launch()
        cmd = call.args[0]

        self.assertTrue(cmd[0].endswith("gmx_MMPBSA"))
        self.assertEqual(cmd[cmd.index("-i") + 1], "mmpbsa.in")
        self.assertEqual(cmd[cmd.index("-cs") + 1], "md.tpr")
        self.assertEqual(cmd[cmd.index("-ct") + 1], "md.xtc")
        self.assertEqual(cmd[cmd.index("-ci") + 1], "mmpbsa_index.ndx")
        self.assertEqual(cmd[cmd.index("-cp") + 1], "topol.top")
        # gmx select writes receptor first, ligand second, so these are fixed.
        self.assertEqual(cmd[cmd.index("-cg") + 1:cmd.index("-cg") + 3], ["0", "1"])

    def test_it_runs_in_the_job_directory_so_topology_includes_resolve(self):
        """A topology is not self-contained: it #includes ligand_GMX.itp and
        posre.itp from beside itself. Running anywhere else killed the run in
        parmed's preprocessor with "Could not find ligand_GMX.itp"."""
        call, _ = self.launch()

        self.assertEqual(call.kwargs.get("cwd"), os.path.abspath(self.working_directory_path))
        self.assertFalse(os.path.isdir(self.path("mmpbsa")),
                         "inputs must not be copied away from their includes")

    def test_scratch_files_are_hidden_from_the_file_listing(self):
        """Dozens of _GMXMMPBSA_* files land beside the results; they are working
        files, so they are hidden the way GROMACS backups are."""
        for name in ("_GMXMMPBSA_COM.pdb", "_GMXMMPBSA_LIG.pdb", "FINAL_RESULTS_MMPBSA.dat"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        files = complex_workflow.get_files_in_working_directory(self.working_directory_path)
        self.assertIn("FINAL_RESULTS_MMPBSA.dat", files)
        self.assertFalse([name for name in files if name.startswith("_GMXMMPBSA_")], files)

    def test_stdin_is_closed(self):
        """gmx_MMPBSA prompts before overwriting; nobody would see the question."""
        call, _ = self.launch()
        self.assertEqual(call.kwargs.get("stdin"), subprocess.DEVNULL)

    def test_more_than_one_process_goes_through_mpirun(self):
        call, _ = self.launch(processes=4)
        cmd = call.args[0]

        self.assertEqual(cmd[:3], ["mpirun", "-np", "4"])
        self.assertEqual(cmd[-1], "MPI")

    def test_a_single_process_does_not(self):
        cmd = self.launch(processes=1)[0].args[0]
        self.assertNotIn("mpirun", cmd)
        self.assertNotEqual(cmd[-1], "MPI")

    def test_starting_marks_the_state_running_and_flips_the_button(self):
        _, (_, status, state, button) = self.launch()

        self.assertTrue(state["running"])
        self.assertEqual(button["value"], "Stop")
        self.assertIn("started", self.plain_text(status))

    def test_clicking_again_stops_the_run(self):
        state = utils.ProcessStateDict()
        state["running"] = True
        state["proc"] = unittest.mock.MagicMock()

        with unittest.mock.patch.object(complex_workflow, "stop_process_gracefully") as stop:
            files, status, returned_state, button = complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", "protein", "resname LIG", 1, state)

        stop.assert_called_once()
        self.assertFalse(returned_state["running"])
        self.assertIsNone(returned_state["proc"])
        self.assertEqual(button["value"], "Start")
        self.assertIn("stopped", self.plain_text(status))


class MmpbsaMpiEnvironmentTests(WorkingDirectoryTestCase):
    """gmx_MMPBSA imports mpi4py even for a serial run, so MPI must start.

    Its Intel MPI probes for a fast fabric and aborts in the OFI provider when
    there is none, which is every laptop, container and WSL install:
    "Fatal error in PMPI_Init_thread ... MPIDI_OFI_mpi_init_hook".
    """

    def test_shared_memory_is_the_default_fabric(self):
        environment = utils.get_gmx_mmpbsa_environment("/opt/gmxMMPBSA/bin/gmx_MMPBSA")
        self.assertEqual(environment[utils.GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE], "shm")

    def test_an_existing_fabric_choice_is_respected(self):
        """A cluster with a real fabric must be able to say so."""
        with unittest.mock.patch.dict(
                os.environ, {utils.GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE: "shm:ofi"}):
            environment = utils.get_gmx_mmpbsa_environment("/opt/gmxMMPBSA/bin/gmx_MMPBSA")
        self.assertEqual(environment[utils.GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE], "shm:ofi")

    def test_the_installation_bin_goes_first_on_path(self):
        """mpirun is a script that looks up mpiexec.hydra by name."""
        environment = utils.get_gmx_mmpbsa_environment("/opt/gmxMMPBSA/bin/gmx_MMPBSA")
        self.assertTrue(environment["PATH"].startswith("/opt/gmxMMPBSA/bin" + os.pathsep))

    def test_mpirun_is_taken_from_the_same_installation(self):
        """A different mpirun on PATH would launch a different MPI than the one
        gmx_MMPBSA is linked against."""
        with unittest.mock.patch.object(utils, "_is_executable", return_value=True):
            self.assertEqual(utils.get_mpirun_beside("/opt/gmxMMPBSA/bin/gmx_MMPBSA"),
                             "/opt/gmxMMPBSA/bin/mpirun")
        with unittest.mock.patch.object(utils, "_is_executable", return_value=False):
            self.assertEqual(utils.get_mpirun_beside("/opt/gmxMMPBSA/bin/gmx_MMPBSA"), "mpirun")

    def launch(self, processes=1):
        for name in ("md.tpr", "md.xtc", "topol.top", "mmpbsa.in", "mmpbsa_index.ndx"):
            with open(self.path(name), "w") as handle:
                handle.write("x")
        state = utils.ProcessStateDict()
        with unittest.mock.patch.object(complex_workflow, "get_gmx_mmpbsa_executable",
                                        return_value="/opt/gmxMMPBSA/bin/gmx_MMPBSA"), \
                unittest.mock.patch.object(complex_workflow, "_build_mmpbsa_index"), \
                unittest.mock.patch.object(complex_workflow, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(complex_workflow, "threading"):
            fake_subprocess.DEVNULL = subprocess.DEVNULL
            fake_subprocess.STDOUT = subprocess.STDOUT
            complex_workflow.on_run_mmpbsa(
                self.working_directory_path, "md.tpr", "md.xtc", "topol.top", "mmpbsa.in",
                "mmpbsa_index.ndx", "group Protein", "resname LIG", processes, state)
        return fake_subprocess.Popen.call_args

    def test_the_launch_carries_the_fabric_setting(self):
        call = self.launch()
        self.assertEqual(call.kwargs["env"][utils.GMX_MMPBSA_FABRIC_ENVIRONMENT_VARIABLE], "shm")

    def test_multi_process_runs_use_the_bundled_mpirun(self):
        # The fixture path does not exist, so pretend the sibling mpirun is there.
        with unittest.mock.patch.object(utils, "_is_executable", return_value=True):
            cmd = self.launch(processes=4).args[0]

        self.assertEqual(cmd[0], "/opt/gmxMMPBSA/bin/mpirun",
                         "a bare mpirun would be whichever MPI happens to be on PATH")
        self.assertEqual(cmd[1:3], ["-np", "4"])
        self.assertEqual(cmd[-1], "MPI")

    def test_output_is_captured_to_a_log_in_the_job_directory(self):
        """The failure that prompted this was only visible in the server terminal."""
        call = self.launch()
        self.assertEqual(call.kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(os.path.exists(self.path(complex_workflow.MMPBSA_LOG_FILE_NAME)))


class MmpbsaResultsLoadingTests(WorkingDirectoryTestCase):
    def test_loading_before_the_run_finishes_says_so(self):
        files, frame, figure, series, histogram, decomposition, decomposition_figure, status = \
            complex_workflow.on_load_mmpbsa_results(
                self.working_directory_path, "FINAL_RESULTS_MMPBSA.dat", "system.pdb",
                "traj.xtc", "mmpbsa.in")

        self.assertIsNone(frame)
        self.assertIn("not found", self.plain_text(status))

    def write(self, name, content):
        with open(self.path(name), "w") as handle:
            handle.write(content)

    def load(self, structure="system.pdb", trajectory="traj.xtc"):
        return complex_workflow.on_load_mmpbsa_results(
            self.working_directory_path, "FINAL_RESULTS_MMPBSA.dat", structure, trajectory,
            "mmpbsa.in")

    def test_results_are_read_from_the_job_directory(self):
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)

        files, frame, figure, series, histogram, decomposition, decomposition_figure, status = self.load()

        self.assertIn("successfully", self.plain_text(status))
        self.assertAlmostEqual(frame.set_index("Term").loc["ΔTOTAL", "Average (kcal/mol)"],
                               -15.02)
        self.assertIsNotNone(figure)
        self.assertIn("FINAL_RESULTS_MMPBSA.dat", files)

    def test_the_histogram_and_residue_table_come_from_the_companion_files(self):
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)
        self.write("FINAL_RESULTS_MMPBSA.csv", PER_FRAME_CSV)
        self.write("FINAL_DECOMP_MMPBSA.csv", DECOMP_CSV)

        _, _, _, series, histogram, decomposition, decomposition_figure, status = self.load()

        self.assertIsNotNone(histogram)
        self.assertIsNotNone(decomposition_figure)
        self.assertEqual(len(decomposition), 3)
        self.assertIn("successfully", self.plain_text(status))
        # Nothing was missing, so nothing is complained about.
        self.assertNotIn("No FINAL", self.plain_text(status))

    def test_a_run_without_decomposition_still_loads_and_says_why(self):
        """The main result must not be withheld because the extras are absent."""
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)

        _, frame, figure, series, histogram, decomposition, decomposition_figure, status = self.load()

        self.assertIsNotNone(frame)
        self.assertIsNotNone(figure)
        self.assertIsNone(histogram)
        self.assertIsNone(decomposition)
        text = self.plain_text(status)
        self.assertIn("successfully", text)
        self.assertIn("Per-residue decomposition", text)

    def test_the_binding_energy_is_plotted_against_simulation_time(self):
        """Frame number would be meaningless: with interval 100, frame 3 of the
        results is trajectory frame 201, not 3."""
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)
        self.write("FINAL_RESULTS_MMPBSA.csv", PER_FRAME_CSV)
        self.write("mmpbsa.in", "&general\n  startframe = 1,\n  interval = 5,\n/\n")
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=2)
        write_trajectory(structure, self.path("traj.xtc"), n_frames=20)

        _, _, _, series, _, _, _, status = self.load()

        axes = series.axes[0]
        self.assertEqual(axes.get_xlabel(), "Time (ns)")
        self.assertEqual(axes.get_ylabel(), "ΔG binding (kcal/mol)")
        self.assertEqual(len(axes.lines[0].get_xdata()), 3)     # three frames in the fixture
        self.assertNotIn("frame number", self.plain_text(status))

    def test_it_falls_back_to_frame_number_and_says_so(self):
        """A missing trajectory must not cost the plot entirely."""
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)
        self.write("FINAL_RESULTS_MMPBSA.csv", PER_FRAME_CSV)
        self.write("mmpbsa.in", "&general\n  startframe = 1,\n  interval = 1,\n/\n")

        _, _, _, series, _, _, _, status = self.load(trajectory="absent.xtc")

        self.assertIsNotNone(series)
        self.assertEqual(series.axes[0].get_xlabel(), "Frame")
        self.assertIn("frame number", self.plain_text(status))

    def test_the_exported_residue_table_keeps_every_residue(self):
        """The chart shows the strongest few; the CSV is the whole table."""
        self.write("FINAL_RESULTS_MMPBSA.dat", RESULTS_DAT)
        self.write("FINAL_DECOMP_MMPBSA.csv", DECOMP_CSV)

        _, _, _, _, _, decomposition, _, _ = self.load()
        files, status = complex_workflow.on_export_df(
            self.working_directory_path, decomposition, "residues.csv")

        self.assertIn("residues.csv", files)
        with open(self.path("residues.csv")) as handle:
            exported = handle.read()
        self.assertIn("Residue,", exported)
        for residue in ("R:A:TRP:59", "R:A:ARG:195", "L:B:UNK:497"):
            self.assertIn(residue, exported)

    def test_results_from_a_run_made_before_the_move_still_load(self):
        """Earlier runs put their results in an mmpbsa/ subdirectory."""
        os.makedirs(self.path("mmpbsa"), exist_ok=True)
        with open(os.path.join(self.path("mmpbsa"), "FINAL_RESULTS_MMPBSA.dat"), "w") as handle:
            handle.write(RESULTS_DAT)

        _, frame, _, _, _, _, _, status = self.load()

        self.assertIn("successfully", self.plain_text(status))
        self.assertAlmostEqual(frame.set_index("Term").loc["ΔTOTAL", "Average (kcal/mol)"],
                               -15.02)

    def test_a_missing_results_file_points_at_the_run_log(self):
        _, frame, _, _, _, _, _, status = self.load()

        self.assertIsNone(frame)
        self.assertIn(complex_workflow.MMPBSA_LOG_FILE_NAME, self.plain_text(status))


if __name__ == "__main__":
    unittest.main()
