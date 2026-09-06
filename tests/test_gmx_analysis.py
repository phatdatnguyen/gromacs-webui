"""Tests for the gmx-driven analyses: what they ask gmx for, and what comes back."""

from __future__ import annotations

import os
import unittest
import unittest.mock

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
import utils
from .testing_support import WorkingDirectoryTestCase, final_result, streamed


class GmxAnalysisCommandTests(WorkingDirectoryTestCase):
    """The command line, without needing GROMACS installed."""

    def capture(self, module, handler_name, *arguments):
        with unittest.mock.patch.object(module, "run_checked_command") as run, \
                unittest.mock.patch.object(module, "read_xvg") as read:
            read.return_value = {"frame": unittest.mock.MagicMock(), "title": "", "ylabel": ""}
            with unittest.mock.patch.object(module, "make_line_figure"):
                # Drained inside the patches: a generator runs nothing until it is.
                self.streamed = streamed(
                    getattr(module, handler_name)(self.working_directory_path, *arguments))
        return run

    def sasa(self, module, surface="protein", output="", probe=0.14):
        return self.capture(module, "on_analyze_sasa", "md.tpr", "md.xtc", surface, output,
                            probe, "sasa.xvg", "sasa_residue.xvg")

    def gyrate(self, module, selection="protein", mode="mass"):
        return self.capture(module, "on_analyze_gyrate", "md.tpr", "md.xtc", selection, mode,
                            "gyrate.xvg")

    def test_sasa_passes_its_selection_probe_and_outputs(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.sasa(module, surface="protein or resname LIG", probe=0.2).call_args.args[0]

                self.assertEqual(cmd[:2], ["gmx", "sasa"])
                self.assertEqual(cmd[cmd.index("-surface") + 1], "protein or resname LIG")
                self.assertEqual(cmd[cmd.index("-probe") + 1], "0.2")
                self.assertEqual(cmd[cmd.index("-o") + 1], "sasa.xvg")
                self.assertEqual(cmd[cmd.index("-or") + 1], "sasa_residue.xvg")

    def test_sasa_omits_the_output_selection_when_it_is_blank(self):
        """-output is optional; passing an empty string would be a parse error."""
        self.assertNotIn("-output", self.sasa(workflow, output="   ").call_args.args[0])
        self.assertIn("-output", self.sasa(workflow, output="resname LIG").call_args.args[0])

    def test_gyrate_passes_its_selection_and_weighting(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.gyrate(module, selection="backbone", mode="geometry").call_args.args[0]

                self.assertEqual(cmd[:2], ["gmx", "gyrate"])
                self.assertEqual(cmd[cmd.index("-sel") + 1], "backbone")
                self.assertEqual(cmd[cmd.index("-mode") + 1], "geometry")

    def test_every_analysis_runs_in_the_job_directory_with_plain_names(self):
        """Same rule as mdrun: gmx byproducts follow the working directory."""
        for module in (workflow, complex_workflow):
            for run in (self.sasa(module), self.gyrate(module)):
                with self.subTest(module=module.__name__, cmd=run.call_args.args[0][1]):
                    # path_security resolves the callback's directory argument.
                    self.assertEqual(run.call_args.kwargs.get("cwd"),
                                     os.path.abspath(self.working_directory_path))
                    for argument in run.call_args.args[0]:
                        self.assertNotIn("/", argument)

    def test_every_analysis_closes_stdin(self):
        """A gmx analysis missing a selection option falls back to an interactive
        prompt; with a live stdin that wedges the worker thread indefinitely."""
        for module in (workflow, complex_workflow):
            for run in (self.sasa(module), self.gyrate(module)):
                with self.subTest(module=module.__name__, cmd=run.call_args.args[0][1]):
                    self.assertEqual(run.call_args.kwargs.get("stdin_input"), "")

    def test_a_gmx_failure_comes_back_as_a_red_status(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                with unittest.mock.patch.object(module, "run_checked_command",
                                                side_effect=Exception("gmx sasa failed")):
                    result = final_result(module.on_analyze_sasa(
                        self.working_directory_path, "md.tpr", "md.xtc", "protein", "", 0.14,
                        "sasa.xvg", "sasa_residue.xvg"))
                self.assertIsNone(result[1])
                text = self.plain_text(result[-1])
                self.assertIn("Error calculating SASA", text)
                self.assertIn("gmx sasa failed", text)


class PcaCommandTests(WorkingDirectoryTestCase):
    """gmx covar and anaeig are legacy tools that prompt for two groups."""

    def capture(self, module, first=1, second=2, selection="backbone"):
        with unittest.mock.patch.object(module, "run_checked_command") as run, \
                unittest.mock.patch.object(module, "read_xvg") as read, \
                unittest.mock.patch.object(module, "make_scree_figure"), \
                unittest.mock.patch.object(module, "make_scatter_figure"):
            read.return_value = {"frame": unittest.mock.MagicMock(), "title": "",
                                 "xlabel": "", "ylabel": ""}
            streams = streamed(module.on_run_pca(
                self.working_directory_path, "md.tpr", "md.xtc", selection, first, second,
                "pca_index.ndx", "pca_eigenvec.trr", "pca_eigenval.xvg", "pca_2dproj.xvg"))
        return [call.args[0] for call in run.call_args_list], run, streams[-1]

    def test_it_runs_select_then_covar_then_anaeig(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                commands, _, _ = self.capture(module)
                self.assertEqual([cmd[1] for cmd in commands], ["select", "covar", "anaeig"])

    def test_an_index_file_is_built_and_reused_by_both_legacy_tools(self):
        """One group in the index leaves covar and anaeig nothing to prompt about,
        and guarantees they see the same atoms."""
        commands, _, _ = self.capture(workflow, selection="name CA")

        select, covar, anaeig = commands
        self.assertEqual(select[select.index("-select") + 1], "name CA")
        self.assertEqual(select[select.index("-on") + 1], "pca_index.ndx")
        self.assertEqual(covar[covar.index("-n") + 1], "pca_index.ndx")
        self.assertEqual(anaeig[anaeig.index("-n") + 1], "pca_index.ndx")

    def test_the_eigenvector_range_reaches_anaeig(self):
        commands, _, _ = self.capture(workflow, first=2, second=5)

        anaeig = commands[2]
        self.assertEqual(anaeig[anaeig.index("-first") + 1], "2")
        self.assertEqual(anaeig[anaeig.index("-last") + 1], "5")
        self.assertEqual(anaeig[anaeig.index("-2d") + 1], "pca_2dproj.xvg")

    def test_covar_and_anaeig_are_given_the_same_eigenvector_file(self):
        commands, _, _ = self.capture(workflow)
        covar, anaeig = commands[1], commands[2]
        self.assertEqual(covar[covar.index("-v") + 1], anaeig[anaeig.index("-v") + 1])

    def test_every_step_closes_stdin(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                _, run, _ = self.capture(module)
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs.get("stdin_input"), "")
                    self.assertEqual(call.kwargs.get("cwd"),
                                     os.path.abspath(self.working_directory_path))

    def test_a_reversed_eigenvector_range_is_refused_before_gmx_runs(self):
        """Nothing at all should run.

        The range used to be checked between covar and anaeig, so a reversed
        range still paid for the covariance calculation - minutes on a real
        trajectory - and overwrote the eigenvector files before complaining.
        """
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                commands, _, result = self.capture(module, first=3, second=2)

                self.assertEqual(commands, [], "gmx ran despite an impossible range")
                self.assertIn("must be higher", self.plain_text(result[-1]))

    def test_an_empty_file_dropdown_is_reported_not_raised(self):
        """A dropdown holds None until the job has a file of that kind, and None
        in argv raises "expected str instance" out of ' '.join, which the user
        sees as a raw Gradio error instead of a status."""
        for module in (workflow, complex_workflow):
            for handler, arguments in (
                    ("on_analyze_sasa", (None, "md.xtc", "group Protein", "", 0.14,
                                         "sasa.xvg", "res.xvg")),
                    ("on_analyze_gyrate", ("md.tpr", None, "group Protein", "mass", "g.xvg")),
                    ("on_run_pca", (None, "md.xtc", "group Backbone", 1, 2, "i.ndx",
                                    "v.trr", "e.xvg", "p.xvg"))):
                with self.subTest(module=module.__name__, handler=handler):
                    with unittest.mock.patch.object(module, "run_checked_command") as run:
                        result = final_result(
                            getattr(module, handler)(self.working_directory_path, *arguments))

                    run.assert_not_called()
                    text = self.plain_text(result[-1])
                    self.assertIn("Select a file for", text)
                    self.assertIn("File Name", text)


MAKE_NDX_LISTING = """\
  0 System              : 74471 atoms
  1 Protein             :  7652 atoms
  2 Protein-H           :  3938 atoms
  3 C-alpha             :   495 atoms
  4 Backbone            :  1485 atoms
 11 non-Protein         : 66819 atoms
 12 Other               :    74 atoms
 13 UNK                 :    74 atoms
 14 NA                  :    71 atoms
 15 CL                  :    68 atoms
 16 Water               : 66606 atoms
 17 SOL                 : 66606 atoms
 19 Ion                 :   139 atoms
"""


class SelectionCandidateTests(WorkingDirectoryTestCase):
    """When a selection matches nothing, say what the structure does contain.

    A job set up before the ligand was normalised to LIG still has it under its
    original name, and "never matches any atoms" alone does not say which.
    """

    def candidates(self, listing=MAKE_NDX_LISTING):
        completed = unittest.mock.MagicMock(stderr=listing, stdout="")
        with unittest.mock.patch.object(utils, "run_checked_command", return_value=completed):
            return utils.describe_selection_candidates("md_0.tpr", self.working_directory_path)

    def test_the_ligand_is_named_with_its_atom_count(self):
        hint = self.candidates()
        self.assertIn("UNK (74 atoms)", hint)
        self.assertIn("resname UNK", hint)

    def test_standard_groups_and_ions_are_not_offered_as_ligands(self):
        hint = self.candidates()
        for noise in ("Protein", "Backbone", "SOL", "Water", "System", "Other", "Ion"):
            self.assertNotIn(f"{noise} (", hint)
        self.assertNotIn("NA (", hint)
        self.assertNotIn("CL (", hint)

    def test_a_system_with_nothing_unusual_offers_nothing(self):
        listing = "\n".join(line for line in MAKE_NDX_LISTING.splitlines()
                            if " UNK " not in line)
        self.assertEqual(self.candidates(listing), "")

    def test_the_hint_never_replaces_the_original_error(self):
        """It is only ever appended, so a failure to probe cannot hide the cause."""
        with unittest.mock.patch.object(utils, "run_checked_command",
                                        side_effect=Exception("make_ndx unavailable")):
            self.assertEqual(
                utils.describe_selection_candidates("md_0.tpr", self.working_directory_path), "")

    def test_a_failing_selection_gets_the_hint_appended(self):
        completed = unittest.mock.MagicMock(stderr=MAKE_NDX_LISTING, stdout="")
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                with unittest.mock.patch.object(
                        module, "run_checked_command",
                        side_effect=Exception("Selection 'resname LIG' never matches any atoms.")), \
                        unittest.mock.patch.object(utils, "run_checked_command",
                                                   return_value=completed):
                    _, _, _, status = final_result(module.on_analyze_gyrate(
                        self.working_directory_path, "md_0.tpr", "md.xtc", "resname LIG",
                        "mass", "gyrate.xvg"))

                text = self.plain_text(status)
                self.assertIn("never matches any atoms", text)
                self.assertIn("UNK", text)

    def test_an_unrelated_failure_is_left_alone(self):
        """Only selection failures get the extra paragraph."""
        with unittest.mock.patch.object(workflow, "run_checked_command",
                                        side_effect=Exception("Cannot open file md.xtc")), \
                unittest.mock.patch.object(utils, "run_checked_command") as probe:
            _, _, _, status = final_result(workflow.on_analyze_gyrate(
                self.working_directory_path, "md_0.tpr", "md.xtc", "group Protein",
                "mass", "gyrate.xvg"))

        probe.assert_not_called()
        self.assertIn("Cannot open file", self.plain_text(status))


class SelectionDefaultTests(unittest.TestCase):
    """The defaults the UI actually ships, read back off the built Blocks.

    A test that hardcodes the expected string proves nothing about what the user
    sees, so these walk the real components webui builds.
    """

    @staticmethod
    def selection_textboxes():
        import gradio as gr
        import webui

        found = {}
        for block in webui.blocks.blocks.values():
            label = getattr(block, "label", None)
            if isinstance(block, gr.Textbox) and label in (
                    "Surface Selection", "Selection", "Receptor Selection", "Ligand Selection"):
                found.setdefault(label, []).append(block.value)
        return found

    def test_no_shipped_selection_combines_a_bare_word_with_a_boolean(self):
        """"protein or ..." parses as one long index group name and is rejected."""
        for label, values in self.selection_textboxes().items():
            for value in values:
                with self.subTest(label=label, value=value):
                    if " or " in value or " and " in value:
                        self.assertNotRegex(
                            value, r"^\s*(protein|backbone|system|water)\b",
                            "a bare group word before a boolean is swallowed as a group name")

    def test_the_complex_surface_default_covers_the_ligand(self):
        """The surface should be every non-solvent atom, so it must include the
        ligand: defaulting to protein alone silently understates the area."""
        surfaces = self.selection_textboxes()["Surface Selection"]
        self.assertTrue(any("LIG" in value for value in surfaces),
                        f"no surface default mentions the ligand: {surfaces}")


class SingleStatusBlockTests(unittest.TestCase):
    """Every analysis reports to the one status block at the top of the tab.

    Walks the real Blocks webui builds, so this checks what is wired rather than
    what the source happens to say.
    """

    @staticmethod
    def analysis_handlers():
        import webui

        wanted = ("on_analyze_", "on_run_pca", "on_run_mmpbsa", "on_load_mmpbsa_results",
                  "on_generate_mmpbsa_input_file")
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        return [h for h in handlers
                if getattr(h.fn, "__name__", "").startswith(wanted)]

    def test_every_analysis_is_wired_to_a_status_output(self):
        handlers = self.analysis_handlers()
        self.assertGreaterEqual(len(handlers), 12, "expected both tabs' analyses")

        import gradio as gr
        for handler in handlers:
            with self.subTest(handler=handler.fn.__name__):
                markdowns = [o for o in handler.outputs if isinstance(o, gr.Markdown)]
                self.assertEqual(len(markdowns), 1,
                                 "an analysis should report to exactly one status block")

    def test_all_of_them_share_the_same_block_per_tab(self):
        """Two blocks in a tab would mean a second status bar had crept back in."""
        import gradio as gr

        blocks_used = set()
        for handler in self.analysis_handlers():
            for output in handler.outputs:
                if isinstance(output, gr.Markdown):
                    blocks_used.add(id(output))

        # One shared status block per tab, and there are two tabs.
        self.assertEqual(len(blocks_used), 2, f"expected 2 status blocks, found {len(blocks_used)}")


class MmpbsaIndexCommandTests(WorkingDirectoryTestCase):
    def test_both_groups_go_into_one_select_option(self):
        """gmx select rejects -select twice: "Option specified multiple times"."""
        with unittest.mock.patch.object(complex_workflow, "run_checked_command") as run, \
                unittest.mock.patch("builtins.open",
                                    unittest.mock.mock_open(read_data="[ Protein ]\n1 2\n"
                                                                      "[ resname_LIG ]\n3\n")):
            complex_workflow._build_mmpbsa_index(
                self.working_directory_path, "md.tpr", "group Protein", "resname LIG",
                "mmpbsa_index.ndx")

        cmd = run.call_args.args[0]
        self.assertEqual(cmd.count("-select"), 1)
        self.assertEqual(cmd[cmd.index("-select") + 1], "group Protein; resname LIG")

    def test_the_receptor_comes_first_so_cg_0_1_is_right(self):
        with unittest.mock.patch.object(complex_workflow, "run_checked_command") as run, \
                unittest.mock.patch("builtins.open",
                                    unittest.mock.mock_open(read_data="[ Protein ]\n1 2\n"
                                                                      "[ resname_LIG ]\n3\n")):
            complex_workflow._build_mmpbsa_index(
                self.working_directory_path, "md.tpr", "group Protein", "resname LIG",
                "mmpbsa_index.ndx")

        selection = run.call_args.args[0][run.call_args.args[0].index("-select") + 1]
        self.assertTrue(selection.startswith("group Protein"), selection)


if __name__ == "__main__":
    unittest.main()
