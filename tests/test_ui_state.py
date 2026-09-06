"""Regression tests for state that must not leak between simulation jobs."""

from __future__ import annotations

import os
import inspect
import unittest

import protein_ligand_complex_md_simulation as complex_workflow
import protein_md_simulation as protein_workflow
import utils

from .testing_support import WorkingDirectoryTestCase


class WorkingDirectoryUIResetTests(WorkingDirectoryTestCase):
    def assert_file_browser_is_cleared(self, reset: tuple) -> None:
        """The common prefix covers every action tied to a selected file."""
        self.assertEqual(reset[:3], (None, None, None))
        for index in (3, 4, 9, 11):
            self.assertFalse(reset[index]["interactive"])
        for index in (5, 6, 7, 8, 12):
            self.assertEqual(reset[index]["value"], "")

        editor = reset[10]
        self.assertEqual(editor["label"], "Text File Viewer")
        self.assertEqual(editor["value"], "")
        self.assertFalse(editor["interactive"])

    def test_protein_job_reset_clears_file_actions_and_analysis_results(self):
        reset = protein_workflow.on_reset_working_directory_ui()

        self.assertEqual(len(reset), 33)
        self.assert_file_browser_is_cleared(reset)
        for state_index in (13, 15, 17, 19, 21, 23, 25, 27):
            self.assertIsNone(reset[state_index])
            self.assertIsNone(reset[state_index + 1]["value"])
        for button in reset[-4:]:
            self.assertEqual(button["value"], "Start")
            self.assertEqual(button["variant"], "primary")

    def test_complex_job_reset_clears_file_actions_and_analysis_results(self):
        reset = complex_workflow.on_reset_working_directory_ui()

        self.assertEqual(len(reset), 44)
        self.assert_file_browser_is_cleared(reset)
        for state_index in (13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 37):
            self.assertIsNone(reset[state_index])
        for plot_index in (14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35, 36, 38):
            self.assertIsNone(reset[plot_index]["value"])
        for button in reset[-5:]:
            self.assertEqual(button["value"], "Start")
            self.assertEqual(button["variant"], "primary")

    def test_open_callback_returns_new_job_and_reset_in_one_update(self):
        """Opening and clearing together leaves no interval with a new path and old file."""
        cases = ((protein_workflow, 5), (complex_workflow, 6))
        for module, open_output_count in cases:
            with self.subTest(module=module.__name__):
                result = module.on_open_working_directory_and_reset_ui(
                    self.working_directory_name)
                self.assertEqual(os.path.basename(result[1]), self.working_directory_name)
                self.assertEqual(
                    result[open_output_count:], module.on_reset_working_directory_ui())

    def test_analysis_prefers_the_production_trajectory_when_fit_is_absent(self):
        for name in ("a.xtc", "md.xtc", "z.trr", "md.tpr"):
            with open(self.path(name), "w") as handle:
                handle.write("fixture")

        cases = ((protein_workflow, 31), (complex_workflow, 36))
        for module, analysis_trajectory_index in cases:
            with self.subTest(module=module.__name__):
                values = {
                    name: "unused.txt"
                    for name in inspect.signature(
                        module.on_file_list_change).parameters
                }
                values.update({
                    "working_directory_path": self.working_directory_path,
                    "prod_md_run_input_file_name": "md.tpr",
                    "fit_backbone_output_traj_file_name": "md_fit.xtc",
                })
                update = module.on_file_list_change(
                    **values)[analysis_trajectory_index]

                self.assertEqual(update["value"], "md.xtc")

    def test_rmsd_and_rmsf_ui_callbacks_pass_the_analysis_tpr(self):
        import webui

        config = webui.blocks.get_config_file()
        components = {component["id"]: component
                      for component in config["components"]}
        analyses = [
            dependency for dependency in config["dependencies"]
            if dependency["api_name"].startswith(
                ("on_analyze_rmsd", "on_analyze_rmsf"))
        ]

        self.assertEqual(len(analyses), 4)
        for dependency in analyses:
            self.assertEqual(len(dependency["inputs"]), 4)
            tpr_component = components[dependency["inputs"][-1]]
            self.assertEqual(
                tpr_component["props"]["label"], "Run Input File Name (.tpr)")


class ProcessTimerLifecycleTests(WorkingDirectoryTestCase):
    def completion_state(self) -> utils.ProcessStateDict:
        state = utils.ProcessStateDict()
        with state["lock"]:
            state.update({
                "working_directory": os.path.realpath(self.working_directory_path),
                "completion_status": "Run completed successfully.",
                "completion_color": "green",
                "completion_pending": True,
            })
        return state

    def test_timer_runs_only_for_a_live_process_or_pending_completion(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                state = utils.ProcessStateDict()
                self.assertFalse(module._process_timer_update(state)["active"])

                with state["lock"]:
                    state["running"] = True
                self.assertTrue(module._process_timer_update(state)["active"])

                with state["lock"]:
                    state["running"] = False
                    state["completion_pending"] = True
                self.assertTrue(module._process_timer_update(state)["active"])

    def test_completion_tick_consumes_notice_and_disables_timer(self):
        for module in (protein_workflow, complex_workflow):
            with self.subTest(module=module.__name__):
                state = self.completion_state()
                files, status, button, timer = \
                    module._sync_process_state_with_timer(
                        self.working_directory_path, state)

                self.assertEqual(files, [])
                self.assertIn("completed successfully", status)
                self.assertEqual(button["value"], "Start")
                self.assertFalse(timer["active"])

                shared = module._sync_shared_process_state_with_timer(
                    self.working_directory_path, self.completion_state())
                self.assertEqual(shared[2]["value"], "Start")
                self.assertEqual(shared[3]["value"], "Start")
                self.assertFalse(shared[4]["active"])

    def test_ui_uses_inactive_self_stopping_timers_without_duplicate_pollers(self):
        import webui

        config = webui.blocks.get_config_file()
        components = {component["id"]: component
                      for component in config["components"]}
        timer_ids = {
            component_id for component_id, component in components.items()
            if component["type"] == "timer"
        }

        # Protein has NVT/NPT/production; complex adds MM-PBSA. Initial and
        # continuation production controls deliberately share one poller.
        self.assertEqual(len(timer_ids), 7)
        self.assertTrue(all(
            components[timer_id]["props"]["active"] is False
            for timer_id in timer_ids
        ))

        ticks = [
            dependency for dependency in config["dependencies"]
            if any(target_id in timer_ids and event == "tick"
                   for target_id, event in dependency["targets"])
        ]
        self.assertEqual(len(ticks), len(timer_ids))
        self.assertEqual(
            {target_id for dependency in ticks
             for target_id, event in dependency["targets"]
             if event == "tick"},
            timer_ids,
        )
        self.assertEqual(len({tuple(dependency["inputs"])
                              for dependency in ticks}), len(timer_ids))
        for dependency in ticks:
            timer_id = next(target_id for target_id, event in dependency["targets"]
                            if event == "tick")
            self.assertIn(timer_id, dependency["outputs"])

        activations = [
            dependency for dependency in config["dependencies"]
            if dependency["api_name"].startswith("_process_timer_update")
        ]
        self.assertEqual(len(activations), 9)
        self.assertTrue(all(dependency["queue"] is False
                            for dependency in activations))
        self.assertTrue(all(dependency["outputs"][0] in timer_ids
                            for dependency in activations))


if __name__ == "__main__":
    unittest.main()
