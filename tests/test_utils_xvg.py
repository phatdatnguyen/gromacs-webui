"""Tests for .xvg parsing, gmx group detection and the standalone figure helper."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import utils

# Captured verbatim from GROMACS 2026.3, including the decoration lines that begin
# with "@ legend" but carry no series, and gyrate's forward-slash escapes (gmx sasa
# uses backslashes for the same thing, so both spellings occur in practice).
GYRATE_XVG = textwrap.dedent("""\
    # This file was created Fri Sep  4 13:24:32 2026
    #                      :-) GROMACS - gmx gyrate, 2026.3 (-:
    #
    @    title "Radius of gyration (total and around axes)"
    @    xaxis  label "Time (ps)"
    @    yaxis  label "Radius (nm)"
    @TYPE xy
    @ view 0.15, 0.15, 0.75, 0.85
    @ legend on
    @ legend box on
    @ legend loctype view
    @ legend 0.78, 0.8
    @ legend length 2
    @ s0 legend "Rg"
    @ s1 legend "Rg/sX/N"
    @ s2 legend "Rg/sY/N"
    @ s3 legend "Rg/sZ/N"
          0.000 0.686945 0.068842 0.684555 0.685880
         10.000 0.690120 0.070100 0.688000 0.689000
         20.000 0.684000 0.067900 0.682000 0.683500
    """)

SINGLE_SERIES_XVG = textwrap.dedent("""\
    @    title "Solvent Accessible Surface"
    @    xaxis  label "Time (ps)"
    @    yaxis  label "Area (nm\\S2\\N)"
        0.0000000    58.1234
        1.0000000    57.9876
    """)

# gmx sasa -or with two output groups: one residue column then (average, stddev)
# per group, but only the first pair is given a legend.
RESAREA_XVG = textwrap.dedent("""\
    @    title "Area per residue over the trajectory"
    @    xaxis  label "Residue"
    @    yaxis  label "Area (nm\\S2\\N)"
    @ s0 legend "Average (nm\\S2\\N)"
    @ s1 legend "Standard deviation (nm\\S2\\N)"
           1    1.527    0.000    1.527    0.000
           2    0.901    0.000    0.901    0.000
    """)

NO_HEADER_XVG = "  0.0  1.0  2.0\n  1.0  3.0  4.0\n"

GENION_MENU = textwrap.dedent("""\
    Group     0 (         System) has  3000 elements
    Group     1 (        Protein) has    45 elements
    Group     2 (      Protein-H) has    23 elements
    Group     3 (        C-alpha) has     6 elements
    Group    12 (          Water) has  2955 elements
    Group    13 (            SOL) has  2955 elements
    Group    17 (  Water_and_ions) has  2955 elements
    """)


class ReadXvgTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, content):
        path = os.path.join(self.directory.name, name)
        with open(path, "w") as handle:
            handle.write(content)
        return path

    def test_columns_are_named_from_the_legends(self):
        data = utils.read_xvg(self.write("gyrate.xvg", GYRATE_XVG))

        self.assertEqual(list(data["frame"].columns),
                         ["Time (ps)", "Rg", "Rg/sX/N", "Rg/sY/N", "Rg/sZ/N"])
        self.assertEqual(len(data["frame"]), 3)
        self.assertAlmostEqual(data["frame"]["Rg"].iloc[0], 0.686945)
        self.assertAlmostEqual(data["frame"]["Time (ps)"].iloc[2], 20.0)

    def test_legend_decoration_lines_are_not_mistaken_for_series(self):
        """"@ legend box on" and "@ legend length 2" name no series."""
        columns = list(utils.read_xvg(self.write("gyrate.xvg", GYRATE_XVG))["frame"].columns)
        self.assertNotIn("box on", columns)
        self.assertEqual(len(columns), 5)

    def test_title_and_axis_labels_come_from_the_file(self):
        data = utils.read_xvg(self.write("gyrate.xvg", GYRATE_XVG))

        self.assertEqual(data["title"], "Radius of gyration (total and around axes)")
        self.assertEqual(data["xlabel"], "Time (ps)")
        self.assertEqual(data["ylabel"], "Radius (nm)")

    def test_unlabelled_trailing_columns_get_positional_names(self):
        """gmx sasa -or writes a pair of columns per group but legends only the first."""
        data = utils.read_xvg(self.write("resarea.xvg", RESAREA_XVG))

        self.assertEqual(list(data["frame"].columns),
                         ["Residue", "Average (nm\\S2\\N)", "Standard deviation (nm\\S2\\N)",
                          "y2", "y3"])
        self.assertEqual(data["frame"]["Residue"].tolist(), [1.0, 2.0])
        self.assertAlmostEqual(data["frame"]["Average (nm\\S2\\N)"].iloc[0], 1.527)

    def test_a_lone_series_is_named_from_the_y_axis(self):
        """gmx sasa writes no legend when there is only one output group."""
        data = utils.read_xvg(self.write("area.xvg", SINGLE_SERIES_XVG))

        self.assertEqual(list(data["frame"].columns), ["Time (ps)", "Area (nm\\S2\\N)"])
        self.assertAlmostEqual(data["frame"].iloc[0, 1], 58.1234)

    def test_a_file_written_with_xvg_none_still_parses(self):
        data = utils.read_xvg(self.write("bare.xvg", NO_HEADER_XVG))

        self.assertEqual(list(data["frame"].columns), ["x", "y0", "y1"])
        self.assertEqual(data["title"], "")
        self.assertEqual(len(data["frame"]), 2)

    def test_a_truncated_final_line_is_dropped(self):
        """A killed run leaves a short row; it must not turn the frame into NaNs."""
        data = utils.read_xvg(self.write("cut.xvg", GYRATE_XVG + "   30.0000000    1.5\n"))

        self.assertEqual(len(data["frame"]), 3)
        self.assertFalse(data["frame"].isna().to_numpy().any())

    def test_repeated_legends_are_disambiguated(self):
        """Duplicate names would make a lookup by name return a frame, not a series."""
        repeated = GYRATE_XVG.replace('@ s1 legend "Rg/sX/N"', '@ s1 legend "Rg"')
        data = utils.read_xvg(self.write("dup.xvg", repeated))

        columns = list(data["frame"].columns)
        self.assertEqual(len(columns), len(set(columns)))
        self.assertIn("Rg", columns)
        self.assertIn("Rg (2)", columns)

    def test_a_file_with_no_data_is_rejected_by_name(self):
        path = self.write("empty.xvg", "# only a comment\n@ title \"nothing\"\n")
        with self.assertRaises(ValueError) as caught:
            utils.read_xvg(path)
        self.assertIn("empty.xvg", str(caught.exception))


class GmxGroupTests(unittest.TestCase):
    def test_finds_a_group_by_name(self):
        self.assertEqual(utils.find_gmx_group_number(GENION_MENU, "SOL"), "13")
        self.assertEqual(utils.find_gmx_group_number(GENION_MENU, "C-alpha"), "3")
        self.assertEqual(utils.find_gmx_group_number(GENION_MENU, "System"), "0")

    def test_names_are_matched_whole_not_by_prefix(self):
        """Water and Water_and_ions share a prefix and are different groups."""
        self.assertEqual(utils.find_gmx_group_number(GENION_MENU, "Water"), "12")
        self.assertEqual(utils.find_gmx_group_number(GENION_MENU, "Water_and_ions"), "17")

    def test_an_absent_group_is_reported_as_none(self):
        self.assertIsNone(utils.find_gmx_group_number(GENION_MENU, "LIG"))
        self.assertIsNone(utils.find_gmx_group_number("", "SOL"))

    def test_every_group_can_be_collected_at_once(self):
        groups = utils.parse_gmx_groups(GENION_MENU)
        self.assertEqual(groups["Protein"], "1")
        self.assertEqual(groups["SOL"], "13")
        self.assertEqual(len(groups), 7)


class LineFigureTests(unittest.TestCase):
    def frame(self):
        import pandas as pd
        return pd.DataFrame({"Time (ns)": [0.0, 1.0, 2.0],
                             "Rg": [1.5, 1.6, 1.4],
                             "RgX": [1.2, 1.3, 1.1]})

    def test_returns_a_standalone_figure_not_a_pyplot_one(self):
        """pyplot's current-figure is process-wide; two concurrent analyses would
        draw into each other."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        before = plt.get_fignums()
        figure = utils.make_line_figure(self.frame())

        self.assertIsInstance(figure, Figure)
        self.assertEqual(plt.get_fignums(), before, "the figure was registered with pyplot")

    def test_plots_every_column_but_the_x_axis_by_default(self):
        axes = utils.make_line_figure(self.frame()).axes[0]
        self.assertEqual([line.get_label() for line in axes.lines], ["Rg", "RgX"])
        self.assertEqual(axes.get_xlabel(), "Time (ns)")

    def test_labels_and_title_are_applied(self):
        figure = utils.make_line_figure(self.frame(), y_columns=["Rg"], xlabel="t",
                                        ylabel="nm", title="Gyration")
        axes = figure.axes[0]
        self.assertEqual(axes.get_xlabel(), "t")
        self.assertEqual(axes.get_ylabel(), "nm")
        self.assertEqual(axes.get_title(), "Gyration")

    def test_mean_line_is_drawn_for_a_single_series(self):
        figure = utils.make_line_figure(self.frame(), y_columns=["Rg"], mean_line=True)
        axes = figure.axes[0]
        self.assertEqual(len(axes.lines), 2)
        self.assertAlmostEqual(axes.lines[1].get_ydata()[0], 1.5)

    def test_the_figure_survives_the_encoding_gradio_puts_it_through(self):
        """A bare Figure has no canvas until something asks for one."""
        from gradio import processing_utils
        encoded = processing_utils.encode_plot_to_base64(
            utils.make_line_figure(self.frame()), "webp")
        self.assertTrue(encoded)


if __name__ == "__main__":
    unittest.main()
