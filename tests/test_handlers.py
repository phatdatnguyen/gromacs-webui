"""Tests for the Gradio callbacks that do not need GROMACS.

The callbacks are wrapped by path_security at import time, so these also exercise
that wrapper on the real signatures.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import unittest
import unittest.mock

import MDAnalysis as mda
import pandas as pd

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as workflow
import utils
from .testing_support import WorkingDirectoryTestCase, write_structure_pdb, write_trajectory

UNK_LIGAND_PDB = textwrap.dedent("""\
    HETATM    1  C1  UNK A 901      12.345  23.456  34.567  1.00  0.00           C
    HETATM    2  O1  UNK A 901      13.345  24.456  35.567  1.00  0.00           O
    END
    """)


class WorkingDirectoryCallbackTests(WorkingDirectoryTestCase):
    def test_opening_a_directory_creates_it_and_enables_the_actions(self):
        result = workflow.on_open_working_directory("_unittest_new_job")
        self.addCleanup(lambda: os.rmdir(os.path.join("data", "_unittest_new_job")))

        self.assertEqual(len(result), 5)
        dropdown, path, files, clean_button, upload = result
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(files, [])
        self.assertTrue(path.endswith("_unittest_new_job"))

    def test_directory_names_with_path_components_are_refused(self):
        """The callback warns and opens nothing rather than escaping ./data."""
        for value in ("../escape", "nested/job", "/absolute", "..\\escape"):
            with self.subTest(value=value):
                self.assertEqual(workflow.on_open_working_directory(value),
                                 (None, None, None, None, None))
        self.assertFalse(os.path.exists(os.path.join("data", "..", "escape")))

    def test_blank_directory_name_returns_nothing(self):
        self.assertEqual(workflow.on_open_working_directory("   "), (None, None, None, None, None))

    def test_file_listing_hides_backups_and_zone_identifiers(self):
        for name in ("keep.gro", "#backup.gro.1#", "note.txt:Zone.Identifier"):
            with open(self.path(name), "w") as handle:
                handle.write("x")
        os.mkdir(self.path("subdir"))

        files = workflow.get_files_in_working_directory(self.working_directory_path)
        self.assertIn("keep.gro", files)
        self.assertNotIn("#backup.gro.1#", files)
        self.assertNotIn("subdir", files)
        self.assertFalse([name for name in files if name.endswith("Zone.Identifier")])

    def test_both_tabs_hide_exactly_the_same_files(self):
        """A job directory is browsable from either tab, so the two listings have
        to agree. They drifted once: only the complex tab hid the MM-PBSA scratch
        files, so the same job showed 109 files in one tab and 80 in the other.
        """
        for name in ("protein.gro", "topol.top", "FINAL_RESULTS_MMPBSA.dat",
                     "_GMXMMPBSA_COM.pdb", "_GMXMMPBSA_LIG.prmtop",
                     "#backup.gro.1#", "notes.txt:Zone.Identifier"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        listings = {module.__name__: module.get_files_in_working_directory(
            self.working_directory_path) for module in (workflow, complex_workflow)}

        self.assertEqual(*listings.values(), f"the two tabs disagree: {listings}")
        shown = next(iter(listings.values()))
        self.assertIn("FINAL_RESULTS_MMPBSA.dat", shown)
        self.assertFalse([name for name in shown if name.startswith("_GMXMMPBSA_")], shown)

    def test_missing_directory_lists_nothing(self):
        self.assertEqual(workflow.get_files_in_working_directory("data/_unittest_absent"), [])
        self.assertEqual(workflow.get_files_in_working_directory(None), [])

    def test_file_listing_is_sorted_by_name(self):
        """Every file dropdown is filtered out of this list, so its order is the UI's."""
        for name in ("zulu.gro", "alpha.top", "Bravo.mdp", "charlie.tpr"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                files = module.get_files_in_working_directory(self.working_directory_path)
                self.assertEqual(files, ["alpha.top", "Bravo.mdp", "charlie.tpr", "zulu.gro"])

    def test_working_directories_are_sorted_by_name(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                directories = module.get_working_directories()
                self.assertEqual(directories, sorted(directories, key=str.lower))
                self.assertIn(self.working_directory_name, directories)


class FileActionTests(WorkingDirectoryTestCase):
    def test_deleting_a_file_refreshes_the_listing(self):
        with open(self.path("doomed.gro"), "w") as handle:
            handle.write("x")
        files = workflow.on_delete_file(self.working_directory_path, "doomed.gro")
        self.assertNotIn("doomed.gro", files)

    def test_cleaning_removes_backups_only(self):
        with open(self.path("#topol.top.1#"), "w") as handle:
            handle.write("x")
        with open(self.path("topol.top"), "w") as handle:
            handle.write("x")

        workflow.on_clean_working_directory(self.working_directory_path)
        remaining = os.listdir(self.working_directory_path)
        self.assertIn("topol.top", remaining)
        self.assertNotIn("#topol.top.1#", remaining)

    def test_text_file_round_trip(self):
        with open(self.path("notes.mdp"), "w") as handle:
            handle.write("integrator = steep\n")

        viewer, save_button = workflow.on_view_text_file(self.working_directory_path, "notes.mdp")
        self.assertIn("integrator = steep", viewer["value"])

        workflow.on_save_text_file(self.working_directory_path, "notes.mdp", "integrator = md\n")
        with open(self.path("notes.mdp")) as handle:
            self.assertEqual(handle.read(), "integrator = md\n")

    def test_export_writes_a_csv_into_the_job_directory(self):
        frame = pd.DataFrame({"Time (ns)": [0.0, 0.1], "RMSD": [0.0, 1.5]})
        files, status = workflow.on_export_df(self.working_directory_path, frame, "rmsd.csv")
        self.assertIn("rmsd.csv", files)
        self.assertIn("File exported", self.plain_text(status))
        self.assertTrue(os.path.exists(self.path("rmsd.csv")))


class AccordionAndSelectionTests(unittest.TestCase):
    def test_selecting_a_structure_opens_its_accordion(self):
        button, accordion = workflow.on_selected_structure_file_state_change("protein.gro")
        self.assertTrue(button["interactive"])
        self.assertTrue(accordion["open"])

    def test_deselecting_disables_the_button_without_closing_the_accordion(self):
        """Closing it would collapse a viewer the user is reading."""
        button, accordion = workflow.on_selected_structure_file_state_change(None)
        self.assertFalse(button["interactive"])
        self.assertNotIn("open", accordion)

    def test_text_selection_behaves_the_same_way(self):
        button, accordion = workflow.on_selected_text_file_state_change("topol.top")
        self.assertTrue(button["interactive"])
        self.assertTrue(accordion["open"])
        button, accordion = workflow.on_selected_text_file_state_change(None)
        self.assertFalse(button["interactive"])
        self.assertNotIn("open", accordion)

    def test_ion_method_switch_shows_the_matching_inputs(self):
        concentration, *counts = workflow.on_add_ions_method_change("Concentration")
        self.assertTrue(concentration["visible"])
        self.assertTrue(all(not update["visible"] for update in counts))

        concentration, *counts = workflow.on_add_ions_method_change("Number")
        self.assertFalse(concentration["visible"])
        self.assertTrue(all(update["visible"] for update in counts))

    def test_mdp_type_switch_renames_the_parameter_file(self):
        seed_box, file_name = workflow.on_change_mdp_type("Initial")
        self.assertTrue(seed_box["visible"])
        self.assertEqual(file_name, "md_initial.mdp")

        seed_box, file_name = workflow.on_change_mdp_type("Continuation")
        self.assertFalse(seed_box["visible"])
        self.assertEqual(file_name, "md_continue.mdp")

    def test_button_state_follows_the_process_state(self):
        state = utils.ProcessStateDict()
        self.assertEqual(workflow.sync_button_state(state)["value"], "Start")
        state["running"] = True
        self.assertEqual(workflow.sync_button_state(state)["value"], "Stop")


class MdpCallbackTests(WorkingDirectoryTestCase):
    def test_each_generator_writes_its_file_and_lists_it(self):
        cases = (
            (workflow.on_generate_ions_mdp_file, ("ions.mdp", "CHARMM36"), "ions.mdp"),
            (workflow.on_generate_energy_minimization_mdp_file, ("em.mdp", "CHARMM36"), "em.mdp"),
        )
        for callback, extra, expected in cases:
            with self.subTest(callback=callback.__name__):
                files, status = callback(self.working_directory_path, *extra)
                self.assertIn(expected, files)
                self.assertIn("successfully", self.plain_text(status))
                with open(self.path(expected)) as handle:
                    self.assertIn("force-switch", handle.read())

    def test_nvt_mdp_callback_passes_the_temperature_through(self):
        files, _ = workflow.on_generate_nvt_equilibration_mdp_file(
            self.working_directory_path, 100, 0.002, 305, "nvt.mdp", "AMBER99SB-ILDN")
        self.assertIn("nvt.mdp", files)
        with open(self.path("nvt.mdp")) as handle:
            content = handle.read()
        self.assertIn("ref_t       = 305", content)
        self.assertIn("gen_temp    = 305", content)
        self.assertIn("-DPOSRES", content)


class TrajectoryViewerCallbackTests(WorkingDirectoryTestCase):
    def test_viewing_a_trajectory_returns_an_iframe_and_a_species_legend(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=3, ions={"NA": 2})
        write_trajectory(structure, self.path("traj.xtc"), n_frames=6)

        html, status = workflow.on_view_trajectory(self.working_directory_path, "system.pdb",
                                                   "traj.xtc", "Protein + Ligand + Ions", 200)
        self.assertIn("<iframe", html)
        self.assertIn("protein_md_trajectory_view.html", html)
        text = self.plain_text(status)
        self.assertIn("Showing 6 of 6 frames", text)
        self.assertIn("NA 2", text)
        self.assertTrue(os.path.exists("static/protein_md_trajectory_view.html"))

    def test_missing_selection_warns_instead_of_raising(self):
        self.assertEqual(workflow.on_view_trajectory(self.working_directory_path, None, None,
                                                     "Protein", 200), (None, None))

    def test_unreadable_pair_reports_an_error_status(self):
        write_structure_pdb(self.path("system.pdb"), n_residues=2)
        with open(self.path("broken.xtc"), "wb") as handle:
            handle.write(b"not really an xtc")

        html, status = workflow.on_view_trajectory(self.working_directory_path, "system.pdb",
                                                   "broken.xtc", "Protein", 200)
        self.assertIsNone(html)
        self.assertIn("Error", self.plain_text(status))

    def test_the_two_tabs_write_to_different_static_files(self):
        structure = write_structure_pdb(self.path("system.pdb"), n_residues=3)
        write_trajectory(structure, self.path("traj.xtc"), n_frames=3)

        workflow.on_view_trajectory(self.working_directory_path, "system.pdb", "traj.xtc", "Protein", 200)
        complex_workflow.on_view_trajectory(self.working_directory_path, "system.pdb", "traj.xtc",
                                            "Protein", 200)
        self.assertTrue(os.path.exists("static/protein_md_trajectory.xtc"))
        self.assertTrue(os.path.exists("static/complex_md_trajectory.xtc"))


class FileListChangeTests(WorkingDirectoryTestCase):
    def arguments(self, count):
        # path_security rejects an empty file name, so use a harmless placeholder
        return ["unused.txt"] * count

    def test_dropdowns_point_at_the_files_each_step_produces(self):
        for name in ("protein.pdb", "protein.gro", "topology.top", "em.mdp", "em.tpr", "md.xtc"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        import inspect
        parameters = inspect.signature(workflow.on_file_list_change).parameters
        values = {name: "unused.txt" for name in parameters}
        values["working_directory_path"] = self.working_directory_path
        values["protein_structure_file_name"] = "protein.pdb"
        values["topology_output_file_name"] = "protein.gro"
        values["topology_output_topology_file_name"] = "topology.top"

        result = workflow.on_file_list_change(**values)

        table = result[0]
        self.assertEqual(sorted(table["File"]), sorted(os.listdir(self.working_directory_path)))
        # first dropdown is the topology input, which should follow the uploaded pdb
        self.assertEqual(result[1]["value"], "protein.pdb")
        # the box input follows pdb2gmx's output structure
        self.assertEqual(result[2]["value"], "protein.gro")

    def test_the_results_dropdown_offers_the_dat_files_in_the_job(self):
        """gmx_MMPBSA writes its report as .dat, and a job can hold several."""
        for name in ("FINAL_RESULTS_MMPBSA.dat", "older_run.dat", "notes.txt"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        import inspect
        parameters = inspect.signature(complex_workflow.on_file_list_change).parameters
        values = {name: "unused.txt" for name in parameters}
        values["working_directory_path"] = self.working_directory_path

        results = complex_workflow.on_file_list_change(**values)[-1]
        self.assertEqual(sorted(results["choices"]), ["FINAL_RESULTS_MMPBSA.dat", "older_run.dat"])
        # The name gmx_MMPBSA writes wins when it is there.
        self.assertEqual(results["value"], "FINAL_RESULTS_MMPBSA.dat")

    def test_every_dropdown_is_offered_files_of_its_own_kind(self):
        """Order, not just count.

        The return tuple and the .change() outputs list are matched positionally,
        so inserting an update in the wrong place silently feeds every later
        dropdown its neighbour's files. The count test cannot see that: the
        lengths still agree. This checks each component against what its label
        says it wants.
        """
        import gradio as gr
        import webui

        expected = {
            "Structure File Name": (".pdb", ".gro"),
            "Input Topology File Name": (".top", ".itp"),
            "Parameter File Name": (".mdp",),
            "Run Input File Name": (".tpr",),
            "Run Input File Name (.tpr)": (".tpr",),
            "Checkpoint File Name": (".cpt",),
            "Input Trajectory File Name": (".xtc", ".trr"),
            "Protein Topology File Name": (".top", ".itp"),
            "Ligand Topology File Name": (".top", ".itp"),
            "Results File Name": (".dat",),
        }
        for name in ("protein.pdb", "protein.gro", "topol.top", "ligand.itp", "em.mdp",
                     "md.tpr", "md.cpt", "md.xtc", "md.trr", "FINAL_RESULTS_MMPBSA.dat"):
            with open(self.path(name), "w") as handle:
                handle.write("x")

        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        checked = 0
        for handler in handlers:
            if getattr(handler.fn, "__name__", "") != "on_file_list_change":
                continue
            import inspect
            required = len(inspect.signature(handler.fn).parameters)
            returned = handler.fn(self.working_directory_path, *self.arguments(required - 1))

            for component, update in zip(handler.outputs, returned):
                if not isinstance(component, gr.Dropdown):
                    continue
                suffixes = expected.get(component.label)
                if suffixes is None:
                    continue
                for choice in update["choices"]:
                    name = choice[1] if isinstance(choice, (list, tuple)) else choice
                    with self.subTest(label=component.label, choice=name):
                        self.assertTrue(name.endswith(suffixes),
                                        f"{component.label!r} was offered {name!r}")
                    checked += 1

        self.assertGreater(checked, 40, "expected to check many dropdown choices")

    def test_return_count_matches_the_wired_outputs(self):
        """A mismatch here breaks every file refresh at runtime."""
        import webui
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        for handler in handlers:
            if getattr(handler.fn, "__name__", "") != "on_file_list_change":
                continue
            import inspect
            required = len(inspect.signature(handler.fn).parameters)
            returned = handler.fn(self.working_directory_path, *self.arguments(required - 1))
            self.assertIsInstance(returned, tuple)
            self.assertEqual(len(returned), len(handler.outputs))


class WiringTests(unittest.TestCase):
    """Every handler must return as many values as it is wired to display.

    FileListChangeTests guards on_file_list_change by calling it; this covers the
    rest statically, including the ones that would need a real GROMACS run or a
    launched subprocess to reach. A handler wired to one output too many fails at
    runtime with nothing in the logs to explain it.
    """

    @staticmethod
    def return_widths():
        """{(module, function): number of values returned}, from the source.

        Keyed by module because the two tabs define handlers of the same name
        with different shapes: on_open_working_directory returns five values in
        one tab and six in the other.
        """
        import ast
        import pathlib

        widths = {}
        for name in ("protein_md_simulation.py", "protein_ligand_complex_md_simulation.py"):
            tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("on_"):
                    continue
                found = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return):
                        value = sub.value
                    elif isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Yield):
                        value = sub.value.value
                    else:
                        continue
                    if isinstance(value, ast.Tuple):
                        found.add(len(value.elts))
                    elif value is not None and not isinstance(value, ast.Constant):
                        found.add(1)
                # Only handlers whose exits already agree; the next test covers
                # the ones that do not.
                if len(found) == 1:
                    widths[(name[:-3], node.name)] = found.pop()
        return widths

    def test_every_exit_of_a_handler_returns_the_same_number_of_values(self):
        """An error path one value short breaks only when something goes wrong."""
        import ast
        import pathlib

        for name in ("protein_md_simulation.py", "protein_ligand_complex_md_simulation.py"):
            tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or not node.name.startswith("on_"):
                    continue
                found = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return):
                        value = sub.value
                    elif isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Yield):
                        value = sub.value.value
                    else:
                        continue
                    if isinstance(value, ast.Tuple):
                        found.add(len(value.elts))
                    elif value is not None and not isinstance(value, ast.Constant):
                        found.add(1)
                with self.subTest(module=name, handler=node.name):
                    self.assertLessEqual(len(found), 1,
                                         f"{node.name} exits with {sorted(found)} values")

    def test_every_handler_matches_the_outputs_it_is_wired_to(self):
        import webui

        widths = self.return_widths()
        handlers = (webui.blocks.fns.values() if hasattr(webui.blocks.fns, "values")
                    else webui.blocks.fns)
        checked = 0
        for handler in handlers:
            key = (getattr(handler.fn, "__module__", ""), getattr(handler.fn, "__name__", ""))
            if key not in widths or not handler.outputs:
                continue
            with self.subTest(module=key[0], handler=key[1]):
                self.assertEqual(len(handler.outputs), widths[key])
            checked += 1

        self.assertGreater(checked, 50, "expected to check most of the wired handlers")


class LigandUploadTests(WorkingDirectoryTestCase):
    """The analysis selects the ligand as "resname LIG", so the upload enforces it."""

    def upload(self, content, name="ligand.pdb", residue_name="LIG"):
        source = os.path.join(tempfile.mkdtemp(), name)
        self.addCleanup(shutil.rmtree, os.path.dirname(source), ignore_errors=True)
        with open(source, "w") as handle:
            handle.write(content)
        return complex_workflow.on_upload_ligand_structure_file(
            self.working_directory_path, name, residue_name, source)

    def test_an_unk_ligand_is_stored_as_lig(self):
        files, status = self.upload(UNK_LIGAND_PDB)

        self.assertIn("ligand.pdb", files)
        with open(self.path("ligand.pdb")) as handle:
            stored = handle.read()
        self.assertNotIn("UNK", stored)
        self.assertIn("LIG", stored)

        plain = self.plain_text(status)
        self.assertIn("uploaded successfully", plain)
        self.assertIn("UNK", plain)                    # the rename is reported, not silent

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)

    def test_a_ligand_already_named_lig_uploads_unchanged(self):
        content = UNK_LIGAND_PDB.replace("UNK", "LIG")
        files, status = self.upload(content)

        self.assertIn("ligand.pdb", files)
        with open(self.path("ligand.pdb")) as handle:
            self.assertEqual(handle.read(), content)
        self.assertNotIn("renamed", self.plain_text(status))

    def test_a_ligand_with_an_empty_residue_field_is_named_lig(self):
        """The regression that made data/3BAJ_5a unanalysable.

        Real ligand PDBs come with columns 18-20 blank. Those records were being
        skipped by the guard that leaves a bare "TER" alone, so the file reached
        acpype unnamed, came back as UNK, and "resname LIG" then matched nothing.
        """
        blank = UNK_LIGAND_PDB.replace("UNK", "   ")
        files, status = self.upload(blank)

        with open(self.path("ligand.pdb")) as handle:
            stored = handle.read()
        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertIn("(blank)", self.plain_text(status))
        self.assertNotIn("HETATM    1  C1     ", stored)

    def test_the_named_residue_is_the_one_renamed(self):
        """A file holding more than the ligand: only the named residue moves."""
        mixed = (UNK_LIGAND_PDB.replace("END\n", "")
                 + "HETATM    3  O   HOH A 902      20.000  20.000  20.000  1.00  0.00           O\n"
                 + "END\n")
        files, status = self.upload(mixed, residue_name="UNK")

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertEqual(universe.select_atoms("resname HOH").n_atoms, 1)
        self.assertIn("UNK", self.plain_text(status))

    def test_a_name_absent_from_the_file_falls_back_to_renaming_everything(self):
        """An uploaded ligand file is the ligand, so the default still works when
        the residue is called something the user did not type."""
        files, status = self.upload(UNK_LIGAND_PDB.replace("UNK", "5A "), residue_name="LIG")

        universe = mda.Universe(self.path("ligand.pdb"))
        self.assertEqual(universe.select_atoms("resname LIG").n_atoms, 2)
        self.assertIn("5A", self.plain_text(status))

    def test_a_protein_upload_keeps_its_own_residue_names(self):
        source = write_structure_pdb(os.path.join(tempfile.mkdtemp(), "protein.pdb"))
        self.addCleanup(shutil.rmtree, os.path.dirname(source), ignore_errors=True)

        complex_workflow.on_upload_protein_structure_file(
            self.working_directory_path, "protein.pdb", source)

        with open(self.path("protein.pdb")) as handle:
            self.assertIn("GLY", handle.read())


class EnergyMinimizationHardwareTests(WorkingDirectoryTestCase):
    """GROMACS has no GPU PME for the minimisers, so this step must stay on the CPU."""

    def run_minimization(self, module, use_gpu):
        with unittest.mock.patch.object(module, "run_checked_command") as run:
            module.on_run_energy_minimization(self.working_directory_path, "em.tpr", 1, 4, use_gpu)
        return run.call_args.args[0]

    def test_the_command_never_asks_for_a_gpu(self):
        for module in (workflow, complex_workflow):
            for use_gpu in (True, False):
                with self.subTest(module=module.__name__, use_gpu=use_gpu):
                    cmd = self.run_minimization(module, use_gpu)
                    self.assertNotIn("gpu", cmd)
                    for task in ("-nb", "-pme", "-bonded"):
                        self.assertEqual(cmd[cmd.index(task) + 1], "cpu")


class GpuCheckboxTests(WorkingDirectoryTestCase):
    """Unticking "Use GPU" has to reach the CPU, not just stop asking for the GPU."""

    RUN_HANDLERS = ("on_run_nvt_equilibration", "on_run_npt_equilibration",
                    "on_run_prod_md", "on_continue_prod_md")

    def launch(self, module, handler_name, use_gpu, nnpot=False):
        import inspect
        handler = getattr(module, handler_name)
        arguments = {}
        for name, parameter in inspect.signature(handler).parameters.items():
            if name == "working_directory_path":
                arguments[name] = self.working_directory_path
            elif name.endswith("process_state"):
                arguments[name] = utils.ProcessStateDict()
            elif name == "run_input_file_name":
                arguments[name] = "md.tpr"
            elif name == "checkpoint_file_name":
                arguments[name] = "md.cpt"
            elif name == "use_gpu":
                arguments[name] = use_gpu
            elif name == "prod_md_nnpot_active":
                arguments[name] = nnpot
            elif parameter.annotation is bool:
                arguments[name] = False
            else:
                arguments[name] = 1

        with unittest.mock.patch.object(module, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(module, "threading"):
            handler(**arguments)
            return fake_subprocess.Popen.call_args.args[0]

    def assertTaskAssignment(self, cmd, task, hardware):
        """Fail with the command rather than a ValueError when the option is absent."""
        self.assertIn(task, cmd, f"{task} left on auto, which picks a detected GPU: {cmd}")
        self.assertEqual(cmd[cmd.index(task) + 1], hardware, cmd)

    def test_unticking_the_box_pins_every_run_to_the_cpu(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.RUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd = self.launch(module, handler_name, use_gpu=False)
                    self.assertNotIn("gpu", cmd)
                    for task in ("-nb", "-pme", "-bonded"):
                        self.assertTaskAssignment(cmd, task, "cpu")

    def test_ticking_the_box_still_offloads(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.RUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd = self.launch(module, handler_name, use_gpu=True)
                    self.assertTaskAssignment(cmd, "-nb", "gpu")

    def test_a_neural_network_run_is_left_on_auto_when_the_box_is_ticked(self):
        """Its own offload set is unsafe here, but the model still wants the GPU."""
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.launch(module, "on_run_prod_md", use_gpu=True, nnpot=True)
                self.assertNotIn("-nb", cmd)
                self.assertNotIn("-pme", cmd)

    def test_a_neural_network_run_still_honours_an_unticked_box(self):
        for module in (workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                cmd = self.launch(module, "on_run_prod_md", use_gpu=False, nnpot=True)
                self.assertTaskAssignment(cmd, "-nb", "cpu")


class MdrunWorkingDirectoryTests(WorkingDirectoryTestCase):
    """mdrun dumps step<n>b.pdb / step<n>c.pdb into its own working directory.

    Those names are hardcoded and no flag redirects them, so running from the
    repository root scatters crash dumps beside the source files.
    """

    MDRUN_HANDLERS = ("on_run_energy_minimization", "on_run_nvt_equilibration",
                      "on_run_npt_equilibration", "on_run_prod_md", "on_continue_prod_md")

    def invoke(self, module, handler_name):
        """Call one mdrun handler with placeholder arguments, capturing the launch."""
        import inspect
        handler = getattr(module, handler_name)
        arguments = {}
        for name, parameter in inspect.signature(handler).parameters.items():
            if name == "working_directory_path":
                arguments[name] = self.working_directory_path
            elif name.endswith("process_state"):
                arguments[name] = utils.ProcessStateDict()
            elif name == "run_input_file_name":
                arguments[name] = "md.tpr"
            elif name == "checkpoint_file_name":
                arguments[name] = "md.cpt"
            elif parameter.annotation is bool:
                arguments[name] = True
            else:
                arguments[name] = 1

        with unittest.mock.patch.object(module, "subprocess") as fake_subprocess, \
                unittest.mock.patch.object(module, "run_checked_command") as run, \
                unittest.mock.patch.object(module, "threading"):
            handler(**arguments)
            if run.called:
                return run.call_args.args[0], run.call_args.kwargs.get("cwd")
            call = fake_subprocess.Popen.call_args
            return call.args[0], call.kwargs.get("cwd")

    def test_every_mdrun_launches_inside_the_job_directory(self):
        for module in (workflow, complex_workflow):
            for handler_name in self.MDRUN_HANDLERS:
                with self.subTest(module=module.__name__, handler=handler_name):
                    cmd, cwd = self.invoke(module, handler_name)

                    self.assertEqual(cmd[:2], ["gmx", "mdrun"])
                    # path_security resolves the callback's directory argument.
                    self.assertEqual(cwd, os.path.abspath(self.working_directory_path))
                    # Plain names only: a path here would be resolved against the
                    # job directory a second time and miss.
                    self.assertEqual(cmd[cmd.index("-deffnm") + 1], "md")
                    if "-cpi" in cmd:
                        self.assertEqual(cmd[cmd.index("-cpi") + 1], "md.cpt")


if __name__ == "__main__":
    unittest.main()
